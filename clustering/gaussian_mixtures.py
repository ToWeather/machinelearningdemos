# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_blobs

class GMMClust():
    def __init__(self, n_components=2, max_iter=100, tol=1e-10):
        self.data_set = None
        self.n_components = n_components
        self.pred_label = None
        self.gamma = None
        self.component_prob = None
        self.means = None
        self.covars = None
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def cal_gaussian_prob(x, mean, covar, delta=1e-10):
        n_dim = x.shape[0]
        covar = covar + delta * np.eye(n_dim)
        prob = np.exp(-0.5 * np.dot((x - mean).reshape(1, n_dim),
                                    np.dot(np.linalg.inv(covar),
                                           (x - mean).reshape(n_dim, 1))))
        prob /= np.sqrt(np.linalg.det(covar) * ((2 * np.pi) ** n_dim))
        return prob

    def cal_sample_likelihood(self, i):
        sample_likelihood = sum(self.component_prob[k] *
                                self.cal_gaussian_prob(self.data_set[i],
                                                       self.means[k], self.covars[k])
                                for k in range(self.n_components))
        return sample_likelihood

    def predict(self, data_set):
        self.data_set = data_set
        self.n_samples, self.n_features = self.data_set.shape
        self.pred_label = np.zeros(self.n_samples, dtype=int)
        self.gamma = np.zeros((self.n_samples, self.n_components))

        start_time = time.time()

        # initializing the parameters
        self.component_prob = [1.0 / self.n_components] * self.n_components
        self.means = np.random.rand(self.n_components, self.n_features)
        for i in range(self.n_features):
            dim_min = np.min(self.data_set[:, i])
            dim_max = np.max(self.data_set[:, i])
            self.means[:, i] = dim_min + (dim_max - dim_min) * self.means[:, i]
        self.covars = np.zeros((self.n_components, self.n_features, self.n_features))
        for i in range(self.n_components):
            self.covars[i] = np.eye(self.n_features)

        # running the EM algorithm for GMM
        pre_L = 0
        iter_cnt = 0
        while iter_cnt <= self.max_iter:
            iter_cnt += 1
            crt_L = 0
            # E step
            for i in range(self.n_samples):
                sample_likelihood = self.cal_sample_likelihood(i)
                crt_L += np.log(sample_likelihood)
                for k in range(self.n_components):
                    self.gamma[i, k] = self.component_prob[k] * \
                                       self.cal_gaussian_prob(self.data_set[i],
                                                              self.means[k],
                                                              self.covars[k]) / sample_likelihood
            # M step
            effective_num = np.sum(self.gamma, axis=0)
            for k in range(self.n_components):
                self.means[k] = sum(self.gamma[i, k] * self.data_set[i] for i in range(self.n_samples))
                self.means[k] /= effective_num[k]
                self.covars[k] = sum(self.gamma[i, k] *
                                     np.outer(self.data_set[i] - self.means[k],
                                              self.data_set[i] - self.means[k])
                                     for i in range(self.n_samples))
                self.covars[k] /= effective_num[k]
                self.component_prob[k] = effective_num[k] / self.n_samples

            print("iteration %s, current value of the log likelihood: %.4f" % (iter_cnt, crt_L))

            if abs(crt_L - pre_L) < self.tol:
                break
            pre_L = crt_L

        self.pred_label = np.argmax(self.gamma, axis=1)
        print("total iteration num: %s, final value of the log likelihood: %.4f, "
              "time used: %.4f seconds" % (iter_cnt, crt_L, time.time() - start_time))


    def plot_clustering(self, kind, y=None, title=None):
        if kind == 1:
            y = self.pred_label
        plt.scatter(self.data_set[:, 0], self.data_set[:, 1],
                    c=y, alpha=0.8)
        if kind == 1:
            plt.scatter(self.means[:, 0], self.means[:, 1],
                        c='r', marker='x')
        if title is not None:
            plt.title(title, size=14)
        plt.axis('on')
        plt.tight_layout()


if __name__ == '__main__':
    # generating a dataset
    n_samples = 1500
    centers = [[0, 0], [5, 6], [8, 3.5]]
    cluster_std = [2, 1.0, 0.5]
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std)
    # running the GMM clustering
    gmm_cluster = GMMClust(n_components=3)
    gmm_cluster.predict(X)
    correct_rate = sum(gmm_cluster.pred_label == y) / n_samples
    print("the rate of the correctly predicted samples: %.4f" % correct_rate)
    # plotting the clustering result
    plt.subplot(1, 2, 1)
    gmm_cluster.plot_clustering(kind=0, y=y, title='The Original Dataset')

    plt.subplot(1, 2, 2)
    gmm_cluster.plot_clustering(kind=1, title='GMM Clustering Result')
    plt.show()