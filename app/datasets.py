__author__ = 'Fenno'

from numpy import array, unique
from numpy.random import rand, permutation
from sklearn import datasets


def simpledataset(n=20, spread=0.6):
    seeds = array([(0.5, 0.5), (0.5, 1.5), (1.5, 0.5), (1.5, 1.5)] * n)
    labels = array([0, 1, 2, 3] * n)
    offsets = spread * rand((4*n), 2) - (spread/2)
    return seeds+offsets, labels, 4


def realdataset(dataset=datasets.load_iris):
    data = dataset().data
    target = dataset().target
    shuffle = permutation(len(target))
    n_clusters = len(unique(target))
    return data[shuffle], target[shuffle], n_clusters