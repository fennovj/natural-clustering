__author__ = 'Fenno'

import numpy as np


def centroidscore(datapoint, clusters, pheromone):
    pass


def acoc_cluster(data, n_clusters, R=10, q0=0.0001, n_iter=1000, beta=2, ro=0.1):
    n_samples = np.shape(data)[0]
    pheromone = 0.75 * np.ones((n_samples, n_clusters))
