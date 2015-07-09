# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 13:21:01 2015

@author: Fenno
"""

from numpy import unique, array, mean, sum, shape, argmin
from numpy.random import randint, rand
from scipy.spatial.distance import minkowski

from clustering.kmeans import kMeansCluster, CACluster


def simpleDataset(n = 20, spread = 0.6):
    seeds = array([(0.5, 0.5), (0.5,1.5), (1.5,0.5), (1.5,1.5)]  * n)
    labels = array([0,1,2,3] * n)
    offsets = spread* rand((4*n),2) - (spread/2)
    return seeds+offsets, labels
    

def randomCluster(data, n_clusters):
    """Just assigns every datapoint a random cluster.
    You should at least beat this score to be considered a clustering method"""
    return randint(n_clusters,size=shape(data)[0])    

def getCentroids(data, labels):
    """Returns a tuple of labelnames, and centroids, ordered in the same way"""
    return array([mean(data[labels == name, :],0) for name in unique(labels)])
    
def getLabels(data, centroids, norm = 2):
    """Returns an array of labels, for the centroid that is closest to each datapoint"""
    return array([argmin([minkowski(data[i,:], centroids[c,:],norm) for c in range(shape(centroids)[0])]) for i in range(shape(data)[0])])
    
def score(data, labels = None, centroids = None, norm = 2):
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
        centroids = getCentroids(data, labels)
    if labels is None:
        labels = getLabels(data, centroids, norm)
    distances = 0
    labelnames = unique(labels)
    for i in range(len(labelnames)):
        datac = data[labels==labelnames[i]]
        distances = distances + sum(array([minkowski(datac[j,:], centroids[i,:], norm) for j in range(shape(datac)[0])]))
    return distances / float(len(labels))
    
def shuffle(data, classes = None):
    """ Shuffle the rows of data.	"""    
    from random import shuffle as rshuffle 
    key = array(range(shape(data)[0]))
    rshuffle(key)
    data = data[key, :]
    classes = classes[key]
    return data, classes

if __name__ == '__main__':
    from sklearn import datasets
    iris = datasets.load_iris().data    
    target = datasets.load_iris().target
    iris, target = shuffle(iris, target)    
    
    iris, target = simpleDataset(20,0.6)    
    
    ncluster = 4 #3 for iris, 4 for simple
    
    labels = kMeansCluster(iris, ncluster)
    randomlabels = randomCluster(iris, ncluster)
    calabels = CACluster(iris, ncluster)
    
    
    print "Random score:", score(iris, randomlabels)
    print "Kmeans score:", score(iris, labels)
    print "Target score:", score(iris, target)
    print "CA score:", score(iris, calabels)