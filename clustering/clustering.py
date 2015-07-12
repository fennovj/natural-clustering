__author__ = 'Fenno'


# The overall parent of clustering methods

from numpy import shape
from numpy.random import randint


class Clustering(object):

    # Data is the data, in a nxf matrix
    # n_clusters is the number of desired clusters
    def cluster(self, data, n_clusters):
        pass


class RandomCluster(Clustering):
    def cluster(self, data, n_clusters):
        """Just assigns every datapoint a random cluster.
        You should at least beat this score to be considered a clustering method"""
        return randint(n_clusters, size=shape(data)[0])


class PerfectCluster(Clustering):
    def __init__(self, target):
        self.target = target

    def cluster(self, data, n_clusters):
        return self.target
