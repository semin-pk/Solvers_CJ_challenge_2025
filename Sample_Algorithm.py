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

        self.cooc = None
        self.zone_cooc = None, 
        self.ordered_zone = None
        self.rack_to_zone = None
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

#SKU 클러스터링 N값 엘보우 기법 적용
    '''def solve_storage_location(self) -> None:
        import networkx as nx
        from sklearn.cluster import SpectralClustering
        from sklearn.metrics import silhouette_score
        import numpy as np
        from collections import defaultdict
        from itertools import combinations

        # 1️⃣ SKU 빈도
        freq = self.orders.groupby('SKU_CD').size()
        skus_by_freq = freq.sort_values(ascending=False).index.tolist()

        # 2️⃣ SKU 간 공출현 (Jaccard)
        order_dict = defaultdict(set)
        for ord_no, sku in self.orders[['ORD_NO', 'SKU_CD']].values:
            order_dict[sku].add(ord_no)

        self.cooc = {}
        for a, b in combinations(skus_by_freq, 2):
            inter = len(order_dict[a] & order_dict[b])
            union = len(order_dict[a] | order_dict[b])
            if union > 0:
                jaccard = inter / union
                self.cooc[(a, b)] = jaccard
                self.cooc[(b, a)] = jaccard

        # 3️⃣ SKU 유사도 행렬
        S_sku = np.zeros((len(skus_by_freq), len(skus_by_freq)))
        for i, a in enumerate(skus_by_freq):
            for j, b in enumerate(skus_by_freq):
                S_sku[i, j] = self.cooc.get((a, b), 0)

        # 4️⃣ 엘보우 기법으로 최적 클러스터 수
        max_k = len(skus_by_freq) // self.params.rack_capacity
        if max_k < 2:
            max_k = 2
        max_k = min(max_k, 10)

        best_k, best_score = 2, -1
        for k in range(2, max_k+1):
            clustering = SpectralClustering(
                n_clusters=k,
                affinity='precomputed',
                assign_labels='discretize',
                random_state=0
            )
            labels = clustering.fit_predict(S_sku)
            try:
                score = silhouette_score(S_sku, labels, metric='precomputed')
            except Exception:
                score = -1
            if score > best_score:
                best_k = k
                best_score = score

        n_clusters = best_k

        # 5️⃣ 최적 k로 클러스터링
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels='discretize',
            random_state=0
        )
        sku_cluster_labels = clustering.fit_predict(S_sku)

        sku_to_cluster = {sku: cluster for sku, cluster in zip(skus_by_freq, sku_cluster_labels)}

        # 6️⃣ 군집별 SKU 모음
        cluster_to_skus = defaultdict(list)
        for sku, cluster in sku_to_cluster.items():
            cluster_to_skus[cluster].append(sku)

        # 7️⃣ Zone 정보
        rack_labels = list(self.od_matrix.index[2:])
        zone_labels = list(set(rack_labels))
        zone_dist_start = self.od_matrix.loc[self.start_location, zone_labels].to_dict()
        ordered_zones = sorted(zone_labels, key=lambda z: zone_dist_start[z])

        # Zone ↔ 랙 매핑 및 용량
        zone_to_racks = {z: [z] for z in zone_labels}
        zone_remaining = {
            z: len(racks) * self.params.rack_capacity
            for z, racks in zone_to_racks.items()
        }

        # 공출현 기반 군집 순서
        cluster_graph = nx.Graph()
        cluster_ids = list(cluster_to_skus.keys())
        for c1, c2 in combinations(cluster_ids, 2):
            weight = 0
            for sku1 in cluster_to_skus[c1]:
                for sku2 in cluster_to_skus[c2]:
                    weight += self.cooc.get((sku1, sku2), 0)
            cluster_graph.add_edge(c1, c2, weight=weight)

        cluster_order = sorted(
            cluster_graph.nodes,
            key=lambda c: -sum(attr['weight'] for attr in dict(cluster_graph[c]).values())
        )

        # 8️⃣ SKU → Zone & 랙 매핑 (분할 배정)
        sku_to_loc = {}
        rack_capacity = self.params.rack_capacity

        for c in cluster_order:
            skus_in_cluster = cluster_to_skus[c]
            sku_idx = 0

            for z in ordered_zones:
                if sku_idx >= len(skus_in_cluster):
                    break

                racks_in_zone = zone_to_racks[z]
                racks_in_zone_sorted = sorted(
                    racks_in_zone,
                    key=lambda r: self.od_matrix.loc[self.start_location, r]
                )

                current_rack_idx = 0
                current_rack_fill = 0

                while sku_idx < len(skus_in_cluster) and zone_remaining[z] > 0:
                    if current_rack_fill >= rack_capacity:
                        current_rack_idx += 1
                        current_rack_fill = 0
                    if current_rack_idx >= len(racks_in_zone_sorted):
                        break

                    sku = skus_in_cluster[sku_idx]
                    rack = racks_in_zone_sorted[current_rack_idx]

                    sku_to_loc[sku] = rack
                    current_rack_fill += 1
                    zone_remaining[z] -= 1
                    sku_idx += 1

            if sku_idx < len(skus_in_cluster):
                raise ValueError(
                    f"전체 Zone에 충분한 용량이 없어 cluster {c}의 SKU 일부를 배치하지 못했습니다."
                )

        # 🔟 결과 반영
        self.orders['LOC'] = self.orders['SKU_CD'].map(sku_to_loc)

        # 속성에 추가
        self.ordered_zone = ordered_zones
        self.rack_to_zone = {}
        for zone, racks in zone_to_racks.items():
            for rack in racks:
                self.rack_to_zone[rack] = zone'''



#랙은 고정이므로 레이아웃만을 사용
    '''def solve_storage_location(self) -> None:
        import networkx as nx
        epsilon = 1e-3
        # 1️⃣ SKU 빈도
        freq = self.orders.groupby('SKU_CD').size()
        skus_by_freq = freq.sort_values(ascending=False).index.tolist()

        # 2️⃣ SKU 간 공출현 (Jaccard)
        order_dict = defaultdict(set)
        for ord_no, sku in self.orders[['ORD_NO', 'SKU_CD']].values:
            order_dict[sku].add(ord_no)
        #n_clusters = int(self.orders['NUM_PCS'].sum() / self.params.cart_capacity)
        self.cooc = {}
        for a, b in combinations(skus_by_freq, 2):
            inter = len(order_dict[a] & order_dict[b])
            union = len(order_dict[a] | order_dict[b])
            if union > 0:
                jaccard = inter / union
                self.cooc[(a, b)] = jaccard
                self.cooc[(b, a)] = jaccard

        # 3️⃣ SKU 클러스터링
        n_clusters = int(np.ceil(len(skus_by_freq) / self.params.rack_capacity))
        sku_labels = skus_by_freq
        S_sku = np.zeros((len(sku_labels), len(sku_labels)))
        for i, a in enumerate(sku_labels):
            for j, b in enumerate(sku_labels):
                S_sku[i, j] = self.cooc.get((a, b), epsilon)

        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels='discretize', #Kmean
            random_state=0
        )
        sku_cluster_labels = clustering.fit_predict(S_sku)

        sku_to_cluster = {sku: cluster for sku, cluster in zip(sku_labels, sku_cluster_labels)}

        # 4️⃣ 클러스터별 SKU 모음
        cluster_to_skus = defaultdict(list)
        for sku, cluster in sku_to_cluster.items():
            cluster_to_skus[cluster].append(sku)

        # 5️⃣ Zone 정보
        rack_labels = list(self.od_matrix.index[2:])
        zone_labels = list(set(rack_labels))
        zone_dist_start = self.od_matrix.loc[self.start_location, zone_labels].to_dict()
        ordered_zones = sorted(zone_labels, key=lambda z: zone_dist_start[z])

        # Zone ↔ 랙 매핑 및 용량
        zone_to_racks = {z: [z] for z in zone_labels}
        zone_remaining = {
            z: len(racks) * self.params.rack_capacity
            for z, racks in zone_to_racks.items()
        }

        # 공출현 기반 클러스터 순서
        cluster_graph = nx.Graph()
        for c1, c2 in combinations(range(n_clusters), 2):
            weight = 0
            for sku1 in cluster_to_skus[c1]:
                for sku2 in cluster_to_skus[c2]:
                    weight += self.cooc.get((sku1, sku2), epsilon)
            cluster_graph.add_edge(c1, c2, weight=weight)

        cluster_order = sorted(
            cluster_graph.nodes,
            key=lambda c: -sum(attr['weight'] for attr in dict(cluster_graph[c]).values())
        )

        # SKU → Zone & 랙 매핑 (분할 배정)
        sku_to_loc = {}
        rack_capacity = self.params.rack_capacity

        for c in cluster_order:
            skus_in_cluster = cluster_to_skus[c]
            sku_idx = 0

            for z in ordered_zones:
                if sku_idx >= len(skus_in_cluster):
                    break

                racks_in_zone = zone_to_racks[z]
                racks_in_zone_sorted = sorted(
                    racks_in_zone,
                    key=lambda r: self.od_matrix.loc[self.start_location, r]
                )

                current_rack_idx = 0
                current_rack_fill = 0

                while sku_idx < len(skus_in_cluster) and zone_remaining[z] > 0:
                    if current_rack_fill >= rack_capacity:
                        current_rack_idx += 1
                        current_rack_fill = 0
                    if current_rack_idx >= len(racks_in_zone_sorted):
                        break

                    sku = skus_in_cluster[sku_idx]
                    rack = racks_in_zone_sorted[current_rack_idx]

                    sku_to_loc[sku] = rack
                    current_rack_fill += 1
                    zone_remaining[z] -= 1
                    sku_idx += 1

            if sku_idx < len(skus_in_cluster):
                raise ValueError(
                    f"전체 Zone에 충분한 용량이 없어 cluster {c}의 SKU 일부를 배치하지 못했습니다."
                )

        # 결과 반영
        self.orders['LOC'] = self.orders['SKU_CD'].map(sku_to_loc)

        # 속성에 추가
        self.ordered_zone = ordered_zones
        self.rack_to_zone = {}
        for zone, racks in zone_to_racks.items():
            for rack in racks:
                self.rack_to_zone[rack] = zone

'''
#TSP 추가

    '''def solve_storage_location(self) -> None:
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
        self.cooc = cooc
        self.rack_to_zone = rack_to_zone
        self.ordered_zones = ordered_zones
        self.zone_cooc = zone_cooc
        # 13. Reflect result
        self.orders['LOC'] = self.orders['SKU_CD'].map(sku_to_location)
'''

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

    '''def solve_storage_location(self) -> None:
        import networkx as nx
        from sklearn.cluster import SpectralClustering
        import numpy as np
        from collections import defaultdict
        from itertools import combinations

        # 1️⃣ SKU 빈도
        freq = self.orders.groupby('SKU_CD').size()
        skus_by_freq = freq.sort_values(ascending=False).index.tolist()

        # 2️⃣ SKU 간 공출현 (Jaccard)
        order_dict = defaultdict(set)
        for ord_no, sku in self.orders[['ORD_NO', 'SKU_CD']].values:
            order_dict[sku].add(ord_no)

        self.cooc = {}
        for a, b in combinations(skus_by_freq, 2):
            inter = len(order_dict[a] & order_dict[b])
            union = len(order_dict[a] | order_dict[b])
            if union > 0:
                jaccard = inter / union
                self.cooc[(a, b)] = jaccard
                self.cooc[(b, a)] = jaccard

        # 3️⃣ SKU 클러스터링 (공출현 기반)
        n_clusters = int(np.ceil(len(skus_by_freq) / self.params.rack_capacity))
        S_sku = np.zeros((len(skus_by_freq), len(skus_by_freq)))
        for i, a in enumerate(skus_by_freq):
            for j, b in enumerate(skus_by_freq):
                S_sku[i, j] = self.cooc.get((a, b), 0)

        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels='discretize',
            random_state=0
        )
        sku_cluster_labels = clustering.fit_predict(S_sku)

        sku_to_cluster = {sku: cluster for sku, cluster in zip(skus_by_freq, sku_cluster_labels)}

        # 4️⃣ 클러스터별 SKU와 우선순위 계산
        cluster_to_skus = defaultdict(list)
        cluster_priority = {}
        for sku, cluster in sku_to_cluster.items():
            cluster_to_skus[cluster].append(sku)

        for cluster, skus in cluster_to_skus.items():
            cluster_priority[cluster] = sum(freq[sku] for sku in skus)

        # 5️⃣ Zone 정보
        rack_labels = list(self.od_matrix.index[2:])
        zone_labels = list(set(rack_labels))
        zone_dist_start = self.od_matrix.loc[self.start_location, zone_labels].to_dict()
        ordered_zones = sorted(zone_labels, key=lambda z: zone_dist_start[z])

        # 6️⃣ Zone ↔ 랙 매핑
        zone_to_racks = {z: [z] for z in zone_labels}
        zone_remaining = {
            z: len(racks) * self.params.rack_capacity
            for z, racks in zone_to_racks.items()
        }

        # 7️⃣ 클러스터 우선순위 순으로 정렬
        clusters_ordered = sorted(cluster_to_skus.keys(), key=lambda c: -cluster_priority[c])

        # 🔷 SKU → LOC 할당
        sku_to_loc = {}
        rack_capacity = self.params.rack_capacity
        used_zones = set()

        for c in clusters_ordered:
            skus_in_cluster = cluster_to_skus[c]
            sku_idx = 0
            candidate_zones = ordered_zones[:]  # Zone 후보 리스트

            while sku_idx < len(skus_in_cluster):
                # 아직 사용하지 않은 Zone 또는 남은 용량이 있는 Zone 선택
                best_zone = None
                for z in candidate_zones:
                    if zone_remaining[z] > 0:
                        best_zone = z
                        break
                if best_zone is None:
                    raise ValueError(f"전체 Zone에 충분한 용량이 없어 cluster {c}의 SKU 일부를 배치하지 못했습니다.")

                racks_in_zone = zone_to_racks[best_zone]
                racks_in_zone_sorted = sorted(
                    racks_in_zone,
                    key=lambda r: self.od_matrix.loc[self.start_location, r]
                )

                current_rack_idx = 0
                current_rack_fill = 0

                while sku_idx < len(skus_in_cluster) and zone_remaining[best_zone] > 0:
                    if current_rack_fill >= rack_capacity:
                        current_rack_idx += 1
                        current_rack_fill = 0
                    if current_rack_idx >= len(racks_in_zone_sorted):
                        break

                    sku = skus_in_cluster[sku_idx]
                    rack = racks_in_zone_sorted[current_rack_idx]

                    sku_to_loc[sku] = rack
                    current_rack_fill += 1
                    zone_remaining[best_zone] -= 1
                    sku_idx += 1

                used_zones.add(best_zone)

        # 🔟 결과 반영
        self.orders['LOC'] = self.orders['SKU_CD'].map(sku_to_loc)

        # 속성에 추가
        self.ordered_zone = ordered_zones
        self.rack_to_zone = {}
        for zone, racks in zone_to_racks.items():
            for rack in racks:
                self.rack_to_zone[rack] = zone'''


#주문 빈도 수와 상품 연관성 고려한 SLAP

    def solve_storage_location(self) -> None:
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
        self.cooc = cooc
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
        print(len(remaining))
        # 결과 반영
        self.orders['LOC'] = self.orders['SKU_CD'].map(sku_to_location)


#FIFO(OBSP)
    '''def solve_order_batching(self) -> None:
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

        self.orders['CART_NO'] = self.orders['ORD_NO'].map(order_to_cart)'''
#단순 그리드클러스터링 배치에 관한 OBSP
    def solve_order_batching(self) -> None:
        """Solve OBSP: 주문을 카트에 배치하고, 카트별 ZONE 순서를 기반으로 SEQ 지정"""
        from collections import defaultdict

        # 1️⃣ 주문별 SKU 목록
        order_skus = self.orders.groupby('ORD_NO')['SKU_CD'].apply(list).to_dict()
        order_locs = self.orders.groupby('ORD_NO')['LOC'].apply(list).to_dict()

        # 2️⃣ 주문별 Zone (랙) 집합
        order_zones = {
            ord_no: set(order_locs[ord_no])
            for ord_no in order_locs
        }

        order_ids = list(order_skus.keys())

        # 3️⃣ 주문 간 유사도 계산 (SKU 연관성의 합)
        order_sim = defaultdict(float)
        for i in range(len(order_ids)):
            for j in range(i+1, len(order_ids)):
                o1, o2 = order_ids[i], order_ids[j]
                sim = sum(
                    self.cooc.get((s1, s2), 0)
                    for s1 in order_skus[o1]
                    for s2 in order_skus[o2]
                )
                order_sim[(o1, o2)] = sim
                order_sim[(o2, o1)] = sim

        # 4️⃣ 주문별 유사 이웃 정렬
        neighbors = {o: [] for o in order_ids}
        for (o1, o2), sim in order_sim.items():
            neighbors[o1].append((o2, sim))
        for o in neighbors:
            neighbors[o].sort(key=lambda x: -x[1])  # 유사도 내림차순

        # 5️⃣ Greedy Batching
        assigned = set()
        order_to_cart = {}
        cart_no = 1
        for o in order_ids:
            if o in assigned:
                continue
            batch = [o]
            assigned.add(o)
            for cand, _ in neighbors[o]:
                if len(batch) >= self.params.cart_capacity:
                    break
                if cand not in assigned:
                    batch.append(cand)
                    assigned.add(cand)
            for ord_id in batch:
                order_to_cart[ord_id] = cart_no
            cart_no += 1

        self.orders['CART_NO'] = self.orders['ORD_NO'].map(order_to_cart)

        # 6️⃣ 카트별 Zone 순서로 SEQ 부여
        # (입출고지점에서 가까운 랙 순으로 카트별 SEQ 지정)
        cart_zones = self.orders.groupby('CART_NO')['LOC'].apply(set).to_dict()
        dist_start = self.od_matrix.loc[self.start_location]

        cart_order = sorted(cart_zones.keys(), key=lambda c: min(
            dist_start[loc] for loc in cart_zones[c] if loc in dist_start
        ))
        cart_to_seq = {cart: idx + 1 for idx, cart in enumerate(cart_order)}

        self.orders['SEQ'] = self.orders['CART_NO'].map(cart_to_seq)

#ZONE Spectral clustering
    '''def solve_order_batching(self) -> None:
        """OBSP: 주문 유사도 + ZONE 순서를 고려한 카트 배치 및 순서"""
        from collections import defaultdict
        import numpy as np

        # 1. 주문별 SKU 집합
        order_skus = self.orders.groupby('ORD_NO')['SKU_CD'].apply(list).to_dict()
        order_zones = {}
        for ord_no, skus in order_skus.items():
            zones = {self.rack_to_zone.get(self.orders.loc[self.orders['SKU_CD'] == sku, 'LOC'].iat[0], -1)
                     for sku in skus}
            order_zones[ord_no] = zones

        order_ids = list(order_skus.keys())

        # 2. 주문 간 유사도 계산
        order_sim = defaultdict(float)
        for i in range(len(order_ids)):
            for j in range(i+1, len(order_ids)):
                o1, o2 = order_ids[i], order_ids[j]
                sim = sum(self.cooc.get((s1, s2), 0) for s1 in order_skus[o1] for s2 in order_skus[o2])
                order_sim[(o1, o2)] = sim
                order_sim[(o2, o1)] = sim

        # 3. 주문별 유사 이웃 정렬
        neighbors = {o: [] for o in order_ids}
        for (o1, o2), sim in order_sim.items():
            neighbors[o1].append((o2, sim))
        for o in neighbors:
            neighbors[o].sort(key=lambda x: -x[1])

        # 4. Greedy Batching
        assigned = set()
        order_to_cart = {}
        cart_no = 1
        for o in order_ids:
            if o in assigned:
                continue
            batch = [o]
            assigned.add(o)
            for cand, _ in neighbors[o]:
                if len(batch) >= self.params.cart_capacity:
                    break
                if cand not in assigned:
                    batch.append(cand)
                    assigned.add(cand)
            for ord_id in batch:
                order_to_cart[ord_id] = cart_no
            cart_no += 1
        self.orders['CART_NO'] = self.orders['ORD_NO'].map(order_to_cart)

        # 5. CART_NO별 ZONE 순서 기반 SEQ 부여
        cart_zones = self.orders.groupby('CART_NO')['LOC'].apply(lambda locs:
            set(self.rack_to_zone[l] for l in locs if l in self.rack_to_zone)).to_dict()

        cart_order = sorted(cart_zones.keys(), key=lambda c: [
            self.ordered_zones.index(z) for z in cart_zones[c] if z in self.ordered_zones
        ])
        cart_to_seq = {cart: idx+1 for idx, cart in enumerate(cart_order)}

        self.orders['SEQ'] = self.orders['CART_NO'].map(cart_to_seq)
'''
# zone Spectral clustering x
    '''def solve_order_batching(self) -> None:
        """OBSP: 주문 유사도 + ZONE 순서를 고려한 카트 배치 및 순서"""
        from collections import defaultdict

        # 1. 주문별 SKU 집합
        order_skus = self.orders.groupby('ORD_NO')['SKU_CD'].apply(list).to_dict()
        order_zones = {}
        for ord_no, skus in order_skus.items():
            zones = {self.rack_to_zone.get(self.orders.loc[self.orders['SKU_CD'] == sku, 'LOC'].iat[0], -1)
                     for sku in skus}
            order_zones[ord_no] = zones

        order_ids = list(order_skus.keys())

        # 2. 주문 간 유사도 계산
        order_sim = defaultdict(float)
        for i in range(len(order_ids)):
            for j in range(i+1, len(order_ids)):
                o1, o2 = order_ids[i], order_ids[j]
                sim = sum(self.cooc.get((s1, s2), 0) for s1 in order_skus[o1] for s2 in order_skus[o2])
                order_sim[(o1, o2)] = sim
                order_sim[(o2, o1)] = sim

        # 3. 주문별 유사 이웃 정렬
        neighbors = {o: [] for o in order_ids}
        for (o1, o2), sim in order_sim.items():
            neighbors[o1].append((o2, sim))
        for o in neighbors:
            neighbors[o].sort(key=lambda x: -x[1])

        # 4. Greedy Batching
        assigned = set()
        order_to_cart = {}
        cart_no = 1
        for o in order_ids:
            if o in assigned:
                continue
            batch = [o]
            assigned.add(o)
            for cand, _ in neighbors[o]:
                if len(batch) >= self.params.cart_capacity:
                    break
                if cand not in assigned:
                    batch.append(cand)
                    assigned.add(cand)
            for ord_id in batch:
                order_to_cart[ord_id] = cart_no
            cart_no += 1
        self.orders['CART_NO'] = self.orders['ORD_NO'].map(order_to_cart)

        # 5. CART_NO별 ZONE 순서 기반 SEQ 부여
        cart_zones = self.orders.groupby('CART_NO')['LOC'].apply(lambda locs:
            set(self.rack_to_zone[l] for l in locs if l in self.rack_to_zone)).to_dict()

        cart_order = sorted(cart_zones.keys(), key=lambda c: [
            self.ordered_zone.index(z) for z in cart_zones[c] if z in self.ordered_zone
        ])
        cart_to_seq = {cart: idx+1 for idx, cart in enumerate(cart_order)}

        self.orders['SEQ'] = self.orders['CART_NO'].map(cart_to_seq)
'''

    def solve_picker_routing(self) -> None:
        """Solve Pick Routing Problem (PRP) using simple sequencing"""
        self.orders = self.orders.sort_values(['CART_NO', 'LOC'])
        self.orders['SEQ'] = self.orders.groupby('CART_NO').cumcount() + 1
    

    def solve(self) -> pd.DataFrame:
        """Execute complete warehouse optimization solution"""
        self.solve_storage_location()
        self.solve_order_batching()
        self.solve_picker_routing()
        if self.orders['LOC'].isna().any():
            raise ValueError("LOC에 할당되지 않은 SKU가 있습니다.")
        if self.orders['CART_NO'].isna().any():
            raise ValueError("CART_NO가 지정되지 않았습니다.")
        if self.orders['SEQ'].isna().any():
            raise ValueError("SEQ가 지정되지 않았습니다.")
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