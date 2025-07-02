import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Set
from collections import defaultdict
from itertools import combinations
from sklearn.cluster import SpectralClustering

# FIFO로 처리하는 샘플 알고리즘 입니다.
# def main처럼 데이터프레임 타입으로 결과 리턴해주시면 됩니다. 데이터는 제공한 Sample_OutputData.csv와 동일한 형태로 리턴해주시면 됩니다.

@dataclass
class WarehouseParameters:
    picking_time: float
    walking_time: float
    cart_capacity: int
    rack_capacity: int
    number_pickers: int

class WarehouseSolver:
    def __init__(self, orders: pd.DataFrame, parameters: pd.DataFrame, od_matrix: pd.DataFrame):
        self.orders = orders.copy()
        self.params = self._load_parameters(parameters)
        self.od_matrix = od_matrix
        self.start_location = od_matrix.index[0]
        self.end_location = od_matrix.index[1]
        
        self._initialize_orders()
        self._validate_input()

    def _load_parameters(self, parameters: pd.DataFrame) -> WarehouseParameters:
        get_param = lambda x: parameters.loc[parameters['PARAMETERS'] == x, 'VALUE'].iloc[0]
        return WarehouseParameters(
            picking_time=float(get_param('PT')),
            walking_time=float(get_param('WT')),
            cart_capacity=int(get_param('CAPA')),
            rack_capacity=int(get_param('RK')),
            number_pickers=int(get_param('PK'))
        )

    def _initialize_orders(self) -> None:
        self.orders['LOC'] = pd.NA
        self.orders['LOC'] = self.orders['LOC'].astype(str)
        self.orders['CART_NO'] = pd.NA
        self.orders['SEQ'] = pd.NA

    def _validate_input(self) -> None:
        if self.orders.empty or self.od_matrix.empty:
            raise ValueError("Input data or OD matrix is empty")
        required_columns = {'ORD_NO', 'SKU_CD'}
        if not required_columns.issubset(self.orders.columns):
            raise ValueError(f"Missing required columns: {required_columns - set(self.orders.columns)}")


    def solve_storage_location(self) -> None:
        import networkx as nx
        # 1. SKU frequency
        if 'NUM_PCS' in self.orders.columns:
            freq = self.orders.groupby('SKU_CD')['NUM_PCS'].sum()
        else:
            freq = self.orders.groupby('SKU_CD').size()
        skus_by_freq = freq.sort_values(ascending=False).index.tolist()

        # 2. SKU co-occurrence matrix (Jaccard similarity)
        order_dict = defaultdict(set)
        for ord_no, sku in self.orders[['ORD_NO', 'SKU_CD']].values:
            order_dict[sku].add(ord_no)

        cooc = {}
        for a, b in combinations(skus_by_freq, 2):
            inter = len(order_dict[a] & order_dict[b])
            union = len(order_dict[a] | order_dict[b])
            if union > 0:
                jaccard = inter / union
                cooc[(a, b)] = jaccard
                cooc[(b, a)] = jaccard

        # 3. Rack distance matrix and similarity
        rack_labels = list(self.od_matrix.index[2:])
        D = self.od_matrix.loc[rack_labels, rack_labels].values
        sigma = np.std(D)
        S = np.exp(-D**2 / (2 * sigma**2))

        # 4. Spectral Clustering for rack zones
        n_zones = int(np.ceil(len(skus_by_freq) / self.params.rack_capacity))
        clustering = SpectralClustering(
            n_clusters=n_zones,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=0
        )
        labels = clustering.fit_predict(S)
        zone_to_racks = defaultdict(list)
        rack_to_zone = {}
        for rack, z in zip(rack_labels, labels):
            zone_to_racks[z].append(rack)
            rack_to_zone[rack] = z

        # 5. Zone entrance distance
        dist_start = self.od_matrix.loc[self.start_location, rack_labels]
        zone_dist = {
            z: np.mean([dist_start[r] for r in racks])
            for z, racks in zone_to_racks.items()
        }

        # 6. SKU clustering into rack-sized groups
        assigned = set()
        clusters = []
        for sku in skus_by_freq:
            if sku in assigned:
                continue
            cluster = {sku}
            candidates = [s for s in skus_by_freq if s not in assigned and s != sku]
            candidates.sort(key=lambda x: cooc.get((sku, x), 0), reverse=True)
            for c in candidates:
                if len(cluster) >= self.params.rack_capacity:
                    break
                cluster.add(c)
            assigned |= cluster
            clusters.append(cluster)

        # 7. Assign each SKU to a zone
        sku_to_zone = {}
        for i, cluster in enumerate(clusters):
            best_zone = i % n_zones  # initial naive assignment
            for sku in cluster:
                sku_to_zone[sku] = best_zone

        # 8. Build zone co-occurrence
        zone_cooc = defaultdict(int)
        for _, grp in self.orders.groupby('ORD_NO'):
            zones = {sku_to_zone.get(sku, -1) for sku in grp['SKU_CD'] if sku in sku_to_zone}
            zones = list(zones)
            for i in range(len(zones)):
                for j in range(i+1, len(zones)):
                    z1, z2 = zones[i], zones[j]
                    zone_cooc[(z1, z2)] += 1
                    zone_cooc[(z2, z1)] += 1

        # 9. Zone graph with distance / (1 + co-occurrence) weights
        zone_list = list(zone_to_racks.keys())
        G = nx.Graph()
        for i, zi in enumerate(zone_list):
            for j, zj in enumerate(zone_list):
                if i == j:
                    continue
                dist = abs(zone_dist[zi] - zone_dist[zj])
                cooc_score = zone_cooc.get((zi, zj), 0)
                weight = dist / (1 + cooc_score)
                G.add_edge(zi, zj, weight=weight)

        # 10. TSP for zone ordering
        tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=False)
        ordered_zones = tsp_path

        # 11. Assign racks based on ordered zones
        rack_sequence = []
        for z in ordered_zones:
            racks = sorted(zone_to_racks[z], key=lambda r: dist_start[r])
            rack_sequence.extend(racks)

        # 12. SKU to location
        sku_to_location = {}
        for cluster, rack in zip(clusters, rack_sequence):
            combined = {
                s: 0.6 * freq.get(s, 0) +
                   0.4 * sum(cooc.get((s, t), 0) for t in cluster if t != s)
                for s in cluster
            }
            sorted_skus = sorted(cluster, key=lambda s: combined[s], reverse=True)
            for sku in sorted_skus:
                sku_to_location[sku] = rack

        # 13. Reflect result
        self.orders['LOC'] = self.orders['SKU_CD'].map(sku_to_location)


# ZONE 거리 클러스팅 -> sku 주문빈도 및 연관성 클러스팅 -> ZONE 크기로 sku 더미 생성 -> 그리드 휴리스틱으로 더미끼리의 연관성 파악 -> 연관성 기반 zone 매핑 -> zone 내에 배치 전략 적용
    '''def solve_storage_location(self) -> None:
        """Solve SLAP using:
           - SKU 빈도·연관성 기반 클러스터링
           - Spectral Clustering 으로 랙 Zone 분할
           - 그리디 휴리스틱으로 Zone 순서 결정
           - 랙 내부 정렬(빈도+연관성 기반)"""
        from collections import defaultdict
        from itertools import combinations
        import numpy as np
        from sklearn.cluster import SpectralClustering

        # 1. SKU 출고 빈도 계산
        if 'NUM_PCS' in self.orders.columns:
            freq = self.orders.groupby('SKU_CD')['NUM_PCS'].sum()
        else:
            freq = self.orders.groupby('SKU_CD').size()
        skus_by_freq = freq.sort_values(ascending=False).index.tolist()

        # 2. SKU 공동 주문 연관성 계산
        cooc = defaultdict(int)
        for _, grp in self.orders.groupby('ORD_NO'):
            items = grp['SKU_CD'].tolist()
            for a, b in combinations(items, 2):
                cooc[(a, b)] += 1
                cooc[(b, a)] += 1

        # 3. 랙 위치 및 거리 준비
        rack_labels = list(self.od_matrix.index[2:])  # WP_000x...
        D = self.od_matrix.loc[rack_labels, rack_labels].values
        sigma = np.std(D)
        S = np.exp(-D**2 / (2 * sigma**2))  # 가우시안 유사도

        # 4. Spectral Clustering 으로 랙을 Zone 분할
        n_zones = int(np.ceil(len(skus_by_freq) / self.params.rack_capacity))
        clustering = SpectralClustering(
            n_clusters=n_zones,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=0
        )
        labels = clustering.fit_predict(S)
        zone_to_racks = defaultdict(list)
        rack_to_zone = {}
        for rack, z in zip(rack_labels, labels):
            zone_to_racks[z].append(rack)
            rack_to_zone[rack] = z

        # 5. 초기 Zone 순서: 입구 기준 거리만으로 정렬
        dist_start = self.od_matrix.loc[self.start_location, rack_labels]
        zone_dist = {
            z: np.mean([dist_start[r] for r in racks])
            for z, racks in zone_to_racks.items()
        }
        initial_ordered_zones = sorted(zone_to_racks.keys(), key=lambda z: zone_dist[z])
        initial_rack_sorted = []
        for z in initial_ordered_zones:
            racks = zone_to_racks[z][:]
            racks.sort(key=lambda r: dist_start[r])
            initial_rack_sorted.extend(racks)

        # 6. SKU 클러스터링 (빈도+연관성 기반)
        assigned = set()
        clusters = []
        for sku in skus_by_freq:
            if sku in assigned:
                continue
            cluster = {sku}
            candidates = [s for s in skus_by_freq if s not in assigned]
            candidates.sort(key=lambda x: cooc.get((sku, x), 0), reverse=True)
            for c in candidates:
                if len(cluster) < self.params.rack_capacity:
                    cluster.add(c)
                else:
                    break
            assigned |= cluster
            clusters.append(cluster)

        # 7. 초기 SKU→랙 매핑 및 SKU→Zone 매핑
        sku_to_initial_rack = {}
        for i, cluster in enumerate(clusters):
            rack = initial_rack_sorted[i]
            for sku in cluster:
                sku_to_initial_rack[sku] = rack
        sku_to_zone = {sku: rack_to_zone[r] for sku, r in sku_to_initial_rack.items()}

        # 8. Zone 간 공동 주문 연관성 계산
        zone_cooc = defaultdict(int)
        for _, grp in self.orders.groupby('ORD_NO'):
            zones = {sku_to_zone[sku] for sku in grp['SKU_CD']}
            for z1, z2 in combinations(zones, 2):
                zone_cooc[(z1, z2)] += 1
                zone_cooc[(z2, z1)] += 1

        # 9. 그리디 휴리스틱으로 최종 Zone 순서 결정
        w_dist, w_cooc = 0.5, 0.5
        def zone_score(cur, nxt):
            return w_cooc * zone_cooc.get((cur, nxt), 0) - w_dist * zone_dist[nxt]

        all_zones = list(zone_to_racks.keys())
        current = min(all_zones, key=lambda z: zone_dist[z])  # 입구에 가장 가까운 Zone
        ordered_zones = [current]
        remaining = set(all_zones) - {current}
        while remaining:
            nxt = max(remaining, key=lambda z: zone_score(current, z))
            ordered_zones.append(nxt)
            remaining.remove(nxt)
            current = nxt

        # 10. 최종 rack_sorted 재구성
        final_rack_sorted = []
        for z in ordered_zones:
            racks = zone_to_racks[z][:]
            racks.sort(key=lambda r: dist_start[r])
            final_rack_sorted.extend(racks)

        # 11. 최종 SKU→랙 매핑 (클러스터 단위 + 내부 정렬)
        sku_to_location = {}
        for idx, cluster in enumerate(clusters):
            rack = final_rack_sorted[idx]
            # 랙 내부 정렬: 빈도 70%, 연관성 30%
            combined = {
                s: 0.7 * freq.get(s, 0) + 
                   0.3 * sum(cooc.get((s, t), 0) for t in cluster if t != s)
                for s in cluster
            }
            sorted_skus = sorted(cluster, key=lambda s: combined[s], reverse=True)
            for sku in sorted_skus:
                sku_to_location[sku] = rack

        # 12. 결과 반영
        self.orders['LOC'] = self.orders['SKU_CD'].map(sku_to_location)'''
    
    
    # Re-define fast SLAP solver
    '''def solve_storage_location(self) -> pd.DataFrame:
        import time
        from itertools import combinations
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        start_time = time.time()
        if 'NUM_PCS' in self.orders.columns:
            freq = self.orders.groupby('SKU_CD')['NUM_PCS'].sum()
        else:
            freq = self.orders.groupby('SKU_CD').size()
        skus_by_freq = freq.sort_values(ascending=False).index.tolist()

        cooc = defaultdict(int)
        top_skus = set(skus_by_freq[:200])
        for _, grp in self.orders.groupby('ORD_NO'):
            items = [sku for sku in grp['SKU_CD'] if sku in top_skus]
            for a, b in combinations(items, 2):
                cooc[(a, b)] += 1
                cooc[(b, a)] += 1

        rack_labels = list(self.od_matrix.index[2:])
        D = self.od_matrix.loc[rack_labels, rack_labels].values

        n_zones = min(int(np.ceil(len(skus_by_freq) / self.params.rack_capacity)), 40)
        D_sym = (D + D.T) / 2
        pca = PCA(n_components=10)
        X = pca.fit_transform(D_sym)
        clustering = KMeans(n_clusters=n_zones, random_state=0, n_init=10)
        labels = clustering.fit_predict(X)
        zone_to_racks = defaultdict(list)
        rack_to_zone = {}
        for rack, z in zip(rack_labels, labels):
            zone_to_racks[z].append(rack)
            rack_to_zone[rack] = z

        dist_start = self.od_matrix.loc[self.start_location, rack_labels]
        zone_dist = {
            z: np.mean([dist_start[r] for r in racks])
            for z, racks in zone_to_racks.items()
        }

        initial_ordered_zones = sorted(zone_to_racks.keys(), key=lambda z: zone_dist[z])
        initial_rack_sorted = []
        for z in initial_ordered_zones:
            racks = zone_to_racks[z][:]
            racks.sort(key=lambda r: dist_start[r])
            initial_rack_sorted.extend(racks)

        assigned = set()
        clusters = []
        for sku in skus_by_freq:
            if sku in assigned:
                continue
            cluster = {sku}
            for other in skus_by_freq:
                if other not in assigned and len(cluster) < self.params.rack_capacity:
                    cluster.add(other)
            assigned |= cluster
            clusters.append(cluster)

        sku_to_location = {}
        for idx, cluster in enumerate(clusters):
            if idx >= len(initial_rack_sorted):
                break
            rack = initial_rack_sorted[idx]
            sorted_skus = sorted(
                cluster,
                key=lambda s: 0.7 * freq.get(s, 0) + 0.3 * sum(cooc.get((s, t), 0) for t in cluster if t != s),
                reverse=True
            )
            for sku in sorted_skus:
                sku_to_location[sku] = rack

        self.orders['LOC'] = self.orders['SKU_CD'].map(sku_to_location)

        elapsed_time = time.time() - start_time
        return self.orders'''
    
    

# 주문 빈도수, 상품 연관성, 랙 위치의 대칭성을 고려한 SLAP
# 랙을 zone으로 클러스터링 -> 가까운 zone부터 빈도수 및 연관성
    '''def solve_storage_location(self) -> None:
        #Solve Storage Location Assignment Problem (SLAP) using SKU frequency, co-occurrence clustering, and spectral clustering for rack zones
        from collections import defaultdict
        from itertools import combinations
        import numpy as np
        from sklearn.cluster import SpectralClustering

        # 1. SKU 출고 빈도 계산
        if 'NUM_PCS' in self.orders.columns:
            freq = self.orders.groupby('SKU_CD')['NUM_PCS'].sum()
        else:
            freq = self.orders.groupby('SKU_CD').size()
        skus_by_freq = freq.sort_values(ascending=False).index.tolist()

        # 2. SKU 간 공동 주문 연관성 계산
        cooc = defaultdict(int)
        for _, group in self.orders.groupby('ORD_NO'):
            for a, b in combinations(group['SKU_CD'], 2):
                cooc[(a, b)] += 1
                cooc[(b, a)] += 1

        # 3. 랙 위치 및 거리 행렬 준비
        rack_locations = list(self.od_matrix.index[2:])
        D = self.od_matrix.loc[rack_locations, rack_locations].values
        # 유사도 행렬 생성 (가우시안 커널)
        sigma = np.std(D)
        S = np.exp(-D**2 / (2 * sigma**2))

        # 4. 랙 ZONE 분할 via Spectral Clustering
        n_clusters = int(np.ceil(len(skus_by_freq) / self.params.rack_capacity))
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels='kmeans'
        )
        labels = clustering.fit_predict(S)
        zone_to_racks = {}
        for rack, label in zip(rack_locations, labels):
            zone_to_racks.setdefault(label, []).append(rack)

        # 5. 랙 순서 결정: 존 순서 + 시작 위치 거리 기준 정렬
        dist_start = self.od_matrix.loc[self.start_location, rack_locations]
        rack_sorted = []
        for label in sorted(zone_to_racks.keys()):
            racks = zone_to_racks[label]
            racks.sort(key=lambda r: dist_start[r])
            rack_sorted.extend(racks)

        # 6. SKU 클러스터링 (빈도 + 연관성)
        assigned = set()
        clusters = []
        for sku in skus_by_freq:
            if sku in assigned:
                continue
            cluster = {sku}
            candidates = [s for s in skus_by_freq if s not in assigned]
            candidates.sort(key=lambda x: cooc.get((sku, x), 0), reverse=True)
            for c in candidates:
                if len(cluster) < self.params.rack_capacity:
                    cluster.add(c)
                else:
                    break
            assigned |= cluster
            clusters.append(cluster)

        # 7. SKU 클러스터를 랙에 매핑
        sku_to_location = {}
        for i, cluster in enumerate(clusters):
            rack = rack_sorted[i] if i < len(rack_sorted) else rack_sorted[i % len(rack_sorted)]
            for sku in cluster:
                sku_to_location[sku] = rack

        # 8. 남은 SKU 처리
        remaining = [s for s in skus_by_freq if s not in sku_to_location]
        for idx, sku in enumerate(remaining, start=len(clusters)):
            rack = rack_sorted[idx % len(rack_sorted)]
            sku_to_location[sku] = rack

        # 9. 결과 반영
        self.orders['LOC'] = self.orders['SKU_CD'].map(sku_to_location)'''


#주문 빈도 수와 상품 연관성 고려한 SLAP

    '''def solve_storage_location(self) -> None:
        """Solve Storage Location Assignment Problem (SLAP) using SKU frequency and co-occurrence clustering"""
        from collections import defaultdict
        from itertools import combinations

        # 1. SKU 출고 빈도 계산 (NUM_PCS가 없으면 주문 건수 기준)
        if 'NUM_PCS' in self.orders.columns:
            freq = self.orders.groupby('SKU_CD')['NUM_PCS'].sum()
        else:
            freq = self.orders.groupby('SKU_CD').size()
        skus_by_freq = freq.sort_values(ascending=False).index.tolist()

        # 2. SKU 간 공동 주문 연관성(co-occurrence) 계산
        cooc = defaultdict(int)
        for _, group in self.orders.groupby('ORD_NO'):
            for a, b in combinations(group['SKU_CD'], 2):
                cooc[(a, b)] += 1
                cooc[(b, a)] += 1

        # 3. 시작 지점에서 가까운 랙부터 정렬
        rack_locations = self.od_matrix.index[2:]
        dist_start = self.od_matrix.loc[self.start_location, rack_locations]
        rack_sorted = dist_start.sort_values().index.tolist()

        # 4. 그리디 클러스터링: 가장 연관성이 높은 SKU들을 같은 클러스터(랙)로 묶기
        assigned = set()
        clusters = []
        for sku in skus_by_freq:
            if sku in assigned:
                continue
            cluster = {sku}
            candidates = [s for s in skus_by_freq if s not in assigned]
            # 연관성 높은 순 정렬
            candidates.sort(key=lambda x: cooc.get((sku, x), 0), reverse=True)
            for c in candidates:
                if len(cluster) < self.params.rack_capacity:
                    cluster.add(c)
                else:
                    break
            assigned |= cluster
            clusters.append(cluster)

        # 5. 클러스터를 랙에 매핑
        sku_to_location = {}
        for rack, cluster in zip(rack_sorted, clusters):
            for sku in cluster:
                sku_to_location[sku] = rack

        # 6. 할당되지 않은 남은 SKU 처리 (랜덤 또는 빈도 순)
        remaining = [s for s in skus_by_freq if s not in sku_to_location]
        idx = len(clusters)
        for sku in remaining:
            rack = rack_sorted[idx // self.params.rack_capacity]
            sku_to_location[sku] = rack
            idx += 1

        # 결과 반영
        self.orders['LOC'] = self.orders['SKU_CD'].map(sku_to_location)'''

    def solve_order_batching(self) -> None:
        """Solve Order Batching and Sequencing Problem (OBSP) using FIFO strategy"""
        unique_orders = sorted(self.orders['ORD_NO'].unique())
        num_carts = len(unique_orders) // self.params.cart_capacity + 1
        
        order_to_cart = {}
        for cart_no in range(1, num_carts + 1):
            start_idx = (cart_no - 1) * self.params.cart_capacity
            end_idx = start_idx + self.params.cart_capacity
            cart_orders = unique_orders[start_idx:end_idx]
            for order in cart_orders:
                order_to_cart[order] = cart_no

        self.orders['CART_NO'] = self.orders['ORD_NO'].map(order_to_cart)

    def solve_picker_routing(self) -> None:
        """Solve Pick Routing Problem (PRP) using simple sequencing"""
        self.orders = self.orders.sort_values(['CART_NO', 'LOC'])
        self.orders['SEQ'] = self.orders.groupby('CART_NO').cumcount() + 1

    def solve(self) -> pd.DataFrame:
        """Execute complete warehouse optimization solution"""
        self.solve_storage_location()
        self.solve_order_batching()
        self.solve_picker_routing()
        return self.orders

def main(INPUT: pd.DataFrame, PARAMETER: pd.DataFrame, OD_MATRIX: pd.DataFrame) -> pd.DataFrame:
    solver = WarehouseSolver(INPUT, PARAMETER, OD_MATRIX)
    return solver.solve()

if __name__ == "__main__":
    import time
    test_INPUT = pd.read_csv("./Sample_InputData.csv")
    test_PARAM = pd.read_csv("./Sample_Parameters.csv")
    test_OD = pd.read_csv("Sample_OD_Matrix.csv", index_col=0, header=0)
    start_time = time.time()
    try:
        '''test_INPUT = pd.read_csv("./Sample_InputData.csv")
        test_PARAM = pd.read_csv("./Sample_Parameters.csv")
        test_OD = pd.read_csv("Sample_OD_Matrix.csv", index_col=0, header=0)'''
        print("Data loaded successfully:")
        print(f"- Orders: {test_INPUT.shape}")
        print(f"- Parameters: {test_PARAM.shape}")
        print(f"- OD Matrix: {test_OD.shape}")

        result = main(test_INPUT, test_PARAM, test_OD)
        result.to_csv("Sample_OutputData.csv", index=False)
        print("\nOptimization completed. Results preview:")
        print(result.head())

    except FileNotFoundError as e:
        print(f"Error: Unable to load required files - {str(e)}")
    except (pd.errors.DataError, pd.errors.EmptyDataError) as e:
        print(f"Error: Data validation failed - {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
    
    print("total_time : ", time.time() - start_time)