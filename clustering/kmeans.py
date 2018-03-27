# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import copy
import time

from sklearn.datasets import make_blobs

class KMeansClust():
    def __init__(self, n_clust=2, max_iter=50, tol=1e-10):
        self.data_set = None
        self.centers_his = []
        self.pred_label = None
        self.pred_label_his = []
        self.n_clust = n_clust
        self.max_iter = max_iter
        self.tol = tol

    def predict(self, data_set):
        self.data_set = data_set
        n_samples, n_features = self.data_set.shape
        self.pred_label = np.zeros(n_samples, dtype=int)

        start_time = time.time()

        # initialize the cluster centers
        centers = np.random.rand(self.n_clust, n_features)
        for i in range(n_features):
            dim_min = np.min(self.data_set[:, i])
            dim_max = np.max(self.data_set[:, i])
            centers[:, i] = dim_min + (dim_max - dim_min) * centers[:, i]
        self.centers_his.append(copy.deepcopy(centers))
        self.pred_label_his.append(copy.deepcopy(self.pred_label))

        print("The initializing cluster centers are: %s" % centers)

        # begin iteration
        pre_J = 1e10
        iter_cnt = 0
        while iter_cnt <= self.max_iter:
            iter_cnt += 1
            # E step: assign samples to clusters
            for i in range(n_samples):
                self.pred_label[i] = np.argmin(np.sum((self.data_set[i, :] - centers) ** 2, axis=1))
            # M step: update centers
            for i in range(self.n_clust):
                centers[i] = np.mean(self.data_set[self.pred_label == i], axis=0)
            self.centers_his.append(copy.deepcopy(centers))
            self.pred_label_his.append(copy.deepcopy(self.pred_label))
            # recalculate the objective function J
            crt_J = np.sum((self.data_set - centers[self.pred_label]) ** 2) / n_samples
            print("iteration %s, current value of J: %.4f" % (iter_cnt, crt_J))
            if np.abs(pre_J - crt_J) < self.tol:
                break
            pre_J = crt_J

        print("total iteration num: %s, final value of J: %.4f, time used: %.4f seconds"
                % (iter_cnt, crt_J, time.time() - start_time))

    # Visualize the clustering
    def plot_clustering(self, iter_cnt=-1, title=None):
        if iter_cnt >= len(self.centers_his) or iter_cnt < -1:
            raise Exception("iter_cnt is not valid!")
        plt.scatter(self.data_set[:, 0], self.data_set[:, 1],
                        c=self.pred_label_his[iter_cnt], alpha=0.8)
        plt.scatter(self.centers_his[iter_cnt][:, 0], self.centers_his[iter_cnt][:, 1],
                        c='r', marker='x')
        if title is not None:
            plt.title(title, size=14)
        plt.axis('on')
        plt.tight_layout()


if __name__ == '__main__':
    # Generate sample data
    n_samples = 1500
    centers = [[0, 0], [5, 6], [8, 3.5]]
    cluster_std = [2, 1.0, 0.5]
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std)

    # Run k-means algorithm
    kmeans_cluster = KMeansClust(n_clust=3)
    kmeans_cluster.predict(X)

    # Visualize the initialization and the final result
    plt.subplots(1, 2)

    plt.subplot(1, 2, 1)
    kmeans_cluster.plot_clustering(iter_cnt=0, title='initialization centers')

    plt.subplot(1, 2, 2)
    kmeans_cluster.plot_clustering(iter_cnt=-1, title='k-means clustering result')
    plt.show()
