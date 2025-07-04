import pandas as pd
import numpy as np
from typing import Callable
import random

def evaluate_completion_time(
    main_func: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame],
    input_df: pd.DataFrame,
    parameter_df: pd.DataFrame,
    od_matrix: pd.DataFrame,
    n_trials: int = 30,
    seed: int = 42
) -> float:
    """
    참가자가 작성한 main 함수를 여러 번 실행하고
    그 결과에 대해 평균 Completion Time을 계산합니다.

    Completion Time 추정 방식 (가상):
    각 CART_NO에 대해 start → 각 LOC 순회 → end 거리 + (픽킹 시간 * SKU 수)
    """

    random.seed(seed)
    np.random.seed(seed)
    times = []

    for trial in range(n_trials):
        # 매 반복마다 같은 시드 유지
        np.random.seed(seed + trial)
        random.seed(seed + trial)

        # 참가자 알고리즘 실행
        result_df = main_func(input_df.copy(), parameter_df.copy(), od_matrix.copy())

        # 피킹 시간, 속도 등 파라미터 추출
        pt = parameter_df.loc[parameter_df['PARAMETERS'] == 'PT', 'VALUE'].values[0]
        wt = parameter_df.loc[parameter_df['PARAMETERS'] == 'WT', 'VALUE'].values[0]
        start = od_matrix.index[0]
        end = od_matrix.index[1]

        total_time = 0

        for cart_no, group in result_df.groupby('CART_NO'):
            route = group.sort_values('SEQ')['LOC'].dropna().tolist()
            if not route:
                continue

            # 경로: Start → LOC1 → LOC2 ... → LOCn → End
            path = [start] + route + [end]
            distance = sum(od_matrix.loc[path[i], path[i + 1]] for i in range(len(path) - 1))

            # 이동 시간 = 거리 / 속도
            walking_time = distance / wt

            # 픽킹 시간 = PT * 아이템 수
            picking_time = pt * len(group)

            total_time += walking_time + picking_time

        times.append(total_time)

    avg_time = np.mean(times)
    print(f"평균 Completion Time over {n_trials} trials: {avg_time:.2f}")
    return avg_time

from Sample_Algorithm import main  # 이미 구현한 main 함수
import time
test_INPUT = pd.read_csv("sample_data/InputData.csv")
test_PARAM = pd.read_csv("sample_data/Parameters.csv")
test_OD = pd.read_csv("sample_data/OD_Matrix.csv", index_col=0, header=0)
'''test_INPUT = pd.read_csv("test_data/InputData_Linked_6p.csv")
test_PARAM = pd.read_csv("test_data/Parameters_Linked_6p.csv")
test_OD = pd.read_csv("test_data/OD_Matrix_Linked_6p.csv", index_col=0, header=0)'''
start_time = time.time()
score = evaluate_completion_time(
    main_func=main,
    input_df=test_INPUT,
    parameter_df=test_PARAM,
    od_matrix=test_OD,
    n_trials=30,
    seed=42
)
print("total_time : ", time.time() - start_time)