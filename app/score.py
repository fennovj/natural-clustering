__author__ = 'Fenno'

from numpy import unique, array, mean, sum, shape, argmin
from scipy.spatial.distance import minkowski
from numpy.random import rand


def randfloat(minimum=0.0, maximum=1.0):
    rangev = maximum - minimum
    return (rangev * rand(1, 1)[0][0]) + minimum


def getcentroids(data, labels):
    """Returns a tuple of labelnames, and centroids, ordered in the same way"""
    return array([mean(data[labels == name, :], 0) for name in unique(labels)])


def getlabels(data, centroids, norm=2):
    """Returns an array of labels, for the centroid that is closest to each datapoint"""
    return array([argmin([minkowski(data[i, :], centroids[c, :], norm)
                          for c in range(shape(centroids)[0])]) for i in range(shape(data)[0])])


def score(data, labels=None, centroids=None, norm=2):
    """Given data and labels or centroids, calculate objective
    the data, and either labels or centroids has to be given, the other is None
    If both labels and centroids are None, an error will be thrown
    If neither labels and centroids are None, they will both be used (although this will typically result
       in a worse score than if leaving one of them as None)
    If one is None, the other isn't, the one that is None will be filled in using the other
    norm is the minkowski norm used for computing distances
    """
    assert (labels is not None) or (centroids is not None), "At least one of labels and centroids must be not None"
    if centroids is None:
        centroids = getcentroids(data, labels)
    if labels is None:
        labels = getlabels(data, centroids, norm)
    distances = 0
    labelnames = unique(labels)
    for i in range(len(labelnames)):
        datac = data[labels == labelnames[i]]
        distances = distances + sum(array([minkowski(datac[j, :], centroids[i, :], norm)
                                           for j in range(shape(datac)[0])]))
    return distances / float(len(labels))
