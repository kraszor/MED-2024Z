import numpy as np
from utils import *


class Cluster:
    def __init__(self, idx: int, points: list, heap: list = []):
        self.idx = idx
        self.points = points
        self.heap = heap

    def __str__(self):
        return f"{self.idx=}: {self.points=}"

    def length(self):
        return len(self.points)

    # def build_local_heap(
    #     self, clusters: list, links: np.ndarray, approximation_function, theta: float
    # ) -> None:
    #     normalization_factor = get_normalization_factor(
    #         approximation_function=approximation_function, theta=theta
    #     )

    #     self.heap = [
    #         (
    #             goodness_measure(
    #                 cluster_1=self,
    #                 cluster_2=cluster,
    #                 adjacency_matrix=links,
    #                 normalization_factor=normalization_factor,
    #             ),
    #             cluster,
    #         )
    #         for cluster in clusters
    #         if cluster.idx != self.idx
    #     ]
    #     self.heap = [item for item in self.heap if item[0] > 0]
    #     self.heap.sort(reverse=True, key=lambda x: x[0])

    def build_local_heap(
        self, clusters: list, links: np.ndarray, approximation_function, theta: float
    ) -> None:
        self.heap = []
        for cluster in clusters:
            goodness = goodness_measure(
                cluster_1=self,
                cluster_2=cluster,
                adjacency_matrix=links,
                normalization_factor=get_normalization_factor(
                    approximation_function=approximation_function, theta=theta
                ),
            )
            if cluster.idx != self.idx and goodness > 0:
                self.heap.append((goodness, cluster))
        self.heap.sort(reverse=True, key=lambda x: x[0])

    def merge_clusters(self, cluster, new_idx: int):
        return Cluster(
            idx=new_idx,
            points=self.points + cluster.points,
        )
