# coding: utf-8

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_moons

from shapely.geometry import Point
import rtree

from sklearn.cluster import DBSCAN as DBSCAN_SKLEARN
from sklearn.preprocessing import StandardScaler


UNCLASSIFIED = -2
NOISE = -1


class DBSCAN():
    def __init__(self, min_pts, eps, metric='euclidean', index_flag=True):
        self.min_pts = min_pts
        self.eps = eps
        self.metric = metric
        self.index_flag = index_flag
        self.data_set = None
        self.pred_label = None
        self.core_points = set()

    def predict(self, data_set):
        self.data_set = data_set
        self.n_samples, self.n_features = self.data_set.shape

        self.data_index = None
        self.dist_matrix = None

        start_time = time.time()
        if self.n_features == 2 and self.metric == 'euclidean' \
            and self.index_flag:
            # using Rtree in a certain case to improve the computation efficiency.
            self.construct_index()
        else:
            # otherwise, compute a distance matrix.
            self.cal_dist_matrix()

        self.pred_label = np.array([UNCLASSIFIED] * self.n_samples)

        # starting clustering.
        crt_cluster_label = -1
        for i in range(self.n_samples):
            if self.pred_label[i] == UNCLASSIFIED:
                query_result = self.query_eps_region_data(i)
                if len(query_result) < self.min_pts:
                    self.pred_label[i] = NOISE
                else:
                    crt_cluster_label += 1
                    self.core_points.add(i)
                    for j in query_result:
                        self.pred_label[j] = crt_cluster_label
                    query_result.discard(i)
                    self.generate_cluster_by_seed(query_result, crt_cluster_label)
        print("time used: %.4f seconds" % (time.time() - start_time))

    def construct_index(self):
        self.data_index = rtree.index.Index()
        for i in range(self.n_samples):
            data = self.data_set[i]
            self.data_index.insert(i, (data[0], data[1], data[0], data[1]))

    @staticmethod
    def distance(data1, data2, metric='euclidean'):
        if metric == 'euclidean':
            dist = np.sqrt(np.dot(data1 - data2, data1 - data2))
        elif metric == 'manhattan':
            dist = np.sum(np.abs(data1 - data2))
        elif metric == 'chebyshev':
            dist = np.max(np.abs(data1 - data2))
        else:
            raise Exception("invalid or unsupported distance metric!")
        return dist

    def cal_dist_matrix(self):
        self.dist_matrix = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            for j in range(i + 1, self.n_samples):
                dist = self.distance(self.data_set[i], self.data_set[j], self.metric)
                self.dist_matrix[i, j], self.dist_matrix[j, i] = dist, dist

    def query_eps_region_data(self, i):
        if self.data_index:
            data = self.data_set[i]
            query_result = set()
            buff_polygon = Point(data[0], data[1]).buffer(self.eps)
            xmin, ymin, xmax, ymax = buff_polygon.bounds
            # query_result = set(self.data_index.intersection((xmin, ymin, xmax, ymax)))
            for idx in self.data_index.intersection((xmin, ymin, xmax, ymax)):
                if Point(self.data_set[idx][0], self.data_set[idx][1]).intersects(buff_polygon):
                    query_result.add(idx)
        else:
            query_result = set(item[0] for item in np.argwhere(self.dist_matrix[i] <= self.eps))
        return query_result

    def generate_cluster_by_seed(self, seed_set, cluster_label):
        while seed_set:
            crt_data_index = seed_set.pop()
            crt_query_result = self.query_eps_region_data(crt_data_index)
            if len(crt_query_result) >= self.min_pts:
                self.core_points.add(crt_data_index)
                for i in crt_query_result:
                    if self.pred_label[i] == UNCLASSIFIED:
                        seed_set.add(i)
                    self.pred_label[i] = cluster_label


def plot_clustering(X, y, core_pts_idx=None, title=None):
    if core_pts_idx is not None:
        core_pts_idx = np.array(list(core_pts_idx), dtype=int)
        core_sample_mask = np.zeros_like(y, dtype=bool)
        core_sample_mask[core_pts_idx] = True

        unique_labels = set(y)
        colors = [plt.cm.Spectral(item) for item in np.linspace(0, 1, len(unique_labels))]

        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = [0, 0, 0, 1]
            class_member_mask = (y == k)
            xy = X[class_member_mask & core_sample_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=12, alpha=0.6)
            xy = X[class_member_mask & ~core_sample_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6, alpha=0.6)
    else:
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.6)
    if title is not None:
        plt.title(title, size=14)
    plt.axis('on')
    plt.tight_layout()


if __name__ == '__main__':
    n_samples = 1500
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)

    dbscan_diy = DBSCAN(min_pts=20, eps=0.5, index_flag=False)
    dbscan_diy.predict(X)

    n_clusters = len(set(dbscan_diy.pred_label)) - (1 if -1 in dbscan_diy.pred_label else 0)
    print("count of clusters generated: %s" % n_clusters)
    print("propotion of noise data for dbscan_diy: %.4f" % (np.sum(dbscan_diy.pred_label == -1) / n_samples))

    plt.subplot(1, 2, 1)
    plot_clustering(X, dbscan_diy.pred_label, dbscan_diy.core_points,
                    title="DBSCAN(DIY) Results")

    dbscan_sklearn = DBSCAN_SKLEARN(min_samples=20, eps=0.5)
    dbscan_sklearn.fit(X)
    print("propotion of noise data for dbscan_sklearn: %.4f" % (np.sum(dbscan_sklearn.labels_ == -1) / n_samples))
    plt.subplot(1, 2, 2)
    plot_clustering(X, dbscan_sklearn.labels_, dbscan_sklearn.core_sample_indices_,
                    title="DBSCAN(SKLEARN) Results")

    plt.show()

    # n_samples = 1500
    # noisy_circles, _ = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    # noisy_circles = StandardScaler().fit_transform(noisy_circles)
    # noisy_moons, _ = make_moons(n_samples=n_samples, noise=.05)
    # noisy_moons = StandardScaler().fit_transform(noisy_moons)
    # dbscan = DBSCAN(min_pts=5, eps=0.22)
    # dbscan.predict(noisy_circles)
    # plt.subplot(1, 2, 1)
    # plot_clustering(noisy_circles, dbscan.pred_label, title="Concentric Circles Dataset")
    #
    # dbscan.predict(noisy_moons)
    # plt.subplot(1, 2, 2)
    # plot_clustering(noisy_moons, dbscan.pred_label, title="Interleaved Moons DataSet")
    #
    # plt.show()
