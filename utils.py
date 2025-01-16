import numpy as np
import os
import pandas as pd
import time


class Point:
    def __init__(
        self,
        init_idx: int,
        idx: int,
        transaction: list,
        cluster_idx: int = None,
        target: str = None,
    ):
        self.init_idx = init_idx
        self.idx = idx
        self.transaction = transaction
        self.cluster_idx = cluster_idx
        self.targer = target


def jaccard_ind(a: list, b: list) -> float:
    a = set(a)
    b = set(b)
    intersection = list(a & b)
    list_union = list(a.union(b))
    indices_a = {item.split(".")[0] for item in a}
    indices_b = {item.split(".")[0] for item in b}
    missing_in_a = indices_b - indices_a
    missing_in_b = indices_a - indices_b
    missing_items_in_a = {item for item in b if item.split(".")[0] in missing_in_a}
    missing_items_in_b = {item for item in a if item.split(".")[0] in missing_in_b}
    to_remove = missing_items_in_a | missing_items_in_b
    for elem in to_remove:
        list_union.remove(elem)
    try:
        to_return = float(len(intersection)) / len(list_union)
    except ZeroDivisionError:
        return 0.0
    return to_return


def goodness_measure(
    cluster_1, cluster_2, adjacency_matrix: list, normalization_factor: float
) -> float:
    start = time.time()
    number_links = calc_num_of_links(cluster_1, cluster_2, adjacency_matrix)
    cluster_1_length = cluster_1.length()
    cluster_2_length = cluster_2.length()
    devider = (
        (cluster_1_length + cluster_2_length) ** normalization_factor
        - cluster_1_length**normalization_factor
        - cluster_2_length**normalization_factor
    )
    result = number_links / devider
    return result


def get_normalization_factor(approximation_function, theta: float) -> float:
    return 1 + 2 * approximation_function(theta)


def calc_num_of_links(cluster_1, cluster_2, adjacency_matrix: list) -> int:
    number_links = 0
    for point_1 in cluster_1.points:
        for point_2 in cluster_2.points:
            number_links += adjacency_matrix[point_1.idx][point_2.idx]
    return number_links


def get_point_neighbors(point: Point, points: list, theta: float) -> list:
    neighbors = []
    for another_point in points:
        if (
            point.idx != another_point.idx
            and jaccard_ind(a=point.transaction, b=another_point.transaction) >= theta
        ):
            neighbors.append(another_point)
    return neighbors


def compute_links(points: list, theta: float) -> np.ndarray:
    nbrlist = [
        get_point_neighbors(point=point, points=points, theta=theta) for point in points
    ]
    n = len(points)
    links = np.zeros(shape=(n, n))
    for i in range(n):
        N = nbrlist[i]
        for j in range(len(N) - 1):
            for l in range(j + 1, len(N)):
                links[N[l].idx][N[j].idx] += 1
                links[N[j].idx][N[l].idx] += 1
    return links


def save_data_to_csv(df, file_name) -> None:
    output_path = os.path.join(os.getcwd(), "results", f"{file_name}.csv")
    df.to_csv(output_path, index=False, header=True)
