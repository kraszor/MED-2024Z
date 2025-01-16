from cluster import Cluster
from utils import (
    compute_links,
    goodness_measure,
    get_normalization_factor,
    Point,
    jaccard_ind,
)
import pandas as pd
import numpy as np


class RockAlgorithm:
    def __init__(
        self,
        data: str,
        k: int,
        theta: float,
        approximation_function,
        drop_outliers: bool = False,
    ) -> None:
        self.data = data
        self.k = k
        self.theta = theta
        self.approximation_function = approximation_function
        self.Q = []
        self.deleted_outliers = []
        self.clusters = []
        self.result = []
        self.drop_outliers = drop_outliers
        self.max_idx = -1
        self.outliers_treshold = 0.3
        self.sample_data = data.get_sample_data()
        self.rest_of_data = data.get_rest_of_data()
        self.links = compute_links(points=self.sample_data, theta=self.theta)
        self.initialization()

    def initialization(self) -> None:
        for point in self.sample_data:
            point_cluster = Cluster(idx=point.idx, points=[point])
            self.max_idx = max(self.max_idx, point.idx)
            self.clusters.append(point_cluster)
        self.n_clusters_start = len(self.clusters)
        self.set_local_heaps()
        self.set_global_heap()
        print("INITIALIZATION DONE", len(self.clusters), len(self.Q))

    def set_local_heaps(self) -> None:
        for cluster in self.clusters:
            cluster.build_local_heap(
                clusters=self.clusters,
                links=self.links,
                approximation_function=self.approximation_function,
                theta=self.theta,
            )

    def set_global_heap(self) -> None:
        self.Q = [
            (cluster.heap[0][0], cluster) for cluster in self.clusters if cluster.heap
        ]

        self.Q.sort(reverse=True, key=lambda x: x[0])

    def get_rid_of_outliers(self) -> None:
        indices_before = set([cluster.idx for cluster in self.clusters])
        self.clusters = [cluster for cluster in self.clusters if cluster.length() > 1]
        self.set_local_heaps()
        self.set_global_heap()
        indices_removed = indices_before - set(
            [cluster.idx for cluster in self.clusters]
        )
        self.deleted_outliers = [
            clustering_point
            for clustering_point in self.sample_data
            if clustering_point.idx in indices_removed
        ]
        print("DROPPED OUTLIERS: ", len(self.deleted_outliers))

    def update_after_merge(
        self,
        cluster_removed_1: Cluster,
        cluster_removed_2: Cluster,
        merged_cluster: Cluster,
    ) -> None:
        heaps = cluster_removed_1.heap + cluster_removed_2.heap
        neighbors = [
            cluster[1]
            for cluster in heaps
            if (
                cluster[1].idx != cluster_removed_1.idx
                and cluster[1].idx != cluster_removed_2.idx
            )
        ]

        normalization_factor = get_normalization_factor(
            approximation_function=self.approximation_function, theta=self.theta
        )

        for cluster in neighbors:
            cluster.heap = [
                cluster_data
                for cluster_data in cluster.heap
                if (
                    cluster_data[1].idx != cluster_removed_1.idx
                    and cluster_data[1].idx != cluster_removed_2.idx
                )
            ]
            merged_goodness = goodness_measure(
                cluster_1=cluster,
                cluster_2=merged_cluster,
                adjacency_matrix=self.links,
                normalization_factor=normalization_factor,
            )
            if merged_goodness > 0:
                cluster.heap.append((merged_goodness, merged_cluster))
                cluster.heap.sort(reverse=True, key=lambda x: x[0])

    def assign_to_cluster(self, clustering_point: Point) -> Cluster:
        best_cluster = None
        best_cluster_score = -1

        for cluster in self.clusters:
            n_neighbors = sum(
                [
                    jaccard_ind(a=clustering_point.transaction, b=neighbour.transaction)
                    >= self.theta
                    for neighbour in cluster.points
                ]
            )
            score = n_neighbors / (
                (cluster.length() + 1) ** self.approximation_function(self.theta)
            )
            if score > best_cluster_score:
                best_cluster_score = score
                best_cluster = cluster

        return best_cluster

    def run(self) -> None:
        flag = False
        while self.Q and len(self.clusters) > self.k:
            u: Cluster = self.Q[0][1]
            v: Cluster = u.heap[0][1]

            self.max_idx += 1
            w: Cluster = u.merge_clusters(v, new_idx=self.max_idx)
            self.clusters.remove(u)
            self.clusters.remove(v)
            self.clusters.append(w)
            w.build_local_heap(
                clusters=self.clusters,
                links=self.links,
                approximation_function=self.approximation_function,
                theta=self.theta,
            )
            self.update_after_merge(
                cluster_removed_1=u, cluster_removed_2=v, merged_cluster=w
            )
            self.set_global_heap()
            if (
                not flag
                and (len(self.clusters) / self.n_clusters_start)
                <= self.outliers_treshold
                and self.drop_outliers
            ):
                self.get_rid_of_outliers()
                flag = True

    def collect_results(self) -> list:
        results = []
        for cluster in self.clusters:
            for point in cluster.points:
                point.cluster_idx = cluster.idx
                results.append(point)
        for point in self.deleted_outliers:
            if not self.drop_outliers:
                predicted_cluster = self.assign_to_cluster(clustering_point=point)
                point.cluster_idx = predicted_cluster.idx
            else:
                point.cluster_idx = None
            results.append(point)
        if self.rest_of_data:
            for point in self.rest_of_data:
                predicted_cluster = self.assign_to_cluster(clustering_point=point)
                point.cluster_idx = predicted_cluster.idx
                results.append(point)
        self.result = results

    def get_rock_output(self, dataset: pd.DataFrame) -> pd.DataFrame:
        sorted_results = sorted(self.result, key=lambda x: x.init_idx)
        print(self.get_silhouette_score(points=self.result))
        cluster_indices = [x.cluster_idx for x in sorted_results]
        cluster_indices = pd.factorize(cluster_indices)[0]
        cluster_indices = pd.Series(cluster_indices, index=dataset.data.index).replace(
            -1, np.nan
        )
        df = pd.DataFrame({"target": dataset.target, "cluster_idx": cluster_indices})
        return df

    def get_silhouette_score(self, points: list) -> float:
        # print(self.result)
        cluster_dict = {}
        for point in points:
            try:
                cluster_dict[point.cluster_idx].append(point)
            except KeyError:
                cluster_dict[point.cluster_idx] = [point]
        silhouettes = []
        for point in points:
            if len(cluster_dict[point.cluster_idx]) == 1:
                continue
            a_i = np.mean(
                [
                    jaccard_ind(
                        a=point.transaction,
                        b=point_2.transaction,
                    )
                    for point_2 in cluster_dict[point.cluster_idx]
                    if point.init_idx != point_2.init_idx
                ]
            )

            b_i = max(
                [
                    np.mean(
                        [
                            jaccard_ind(
                                a=point.transaction,
                                b=point_2.transaction,
                            )
                            for point_2 in cluster_dict[cluster_idx]
                        ]
                    )
                    for cluster_idx in cluster_dict.keys()
                    if cluster_idx != point.cluster_idx
                ]
            )
            silhouettes.append((-b_i + a_i) / max(a_i, b_i))
        return np.mean(np.array(silhouettes))
