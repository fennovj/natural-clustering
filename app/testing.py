# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 13:21:01 2015

@author: Fenno
"""

from numpy import unique, array, mean, sum, shape, argmin
from numpy.random import rand
from scipy.spatial.distance import minkowski
from clustering.clustering import randomCluster, perfectCluster
from clustering.kmeans import kMeansCluster
#from clustering.cellular_automata import import CACluster


def simpleDataset(n = 20, spread = 0.6):
    seeds = array([(0.5, 0.5), (0.5,1.5), (1.5,0.5), (1.5,1.5)]  * n)
    labels = array([0,1,2,3] * n)
    offsets = spread* rand((4*n),2) - (spread/2)
    return seeds+offsets, labels
    
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

def testclusterers(data, n_cluster, **clusterers):

    keys = clusterers.keys()
    labels = [clusterers[key].cluster(data, n_cluster) for key in keys]

    for i, key in enumerate(keys):
        print key, "score:", score(data,labels[i])

if __name__ == '__main__':
    #from sklearn import datasets
    #iris = datasets.load_iris().data
    #target = datasets.load_iris().target
    #iris, target = shuffle(iris, target)
    
    data, target = simpleDataset(20,0.6)
    
    n_cluster = 4 #3 for iris, 4 for simple

    clusterers = {'kMeans' : kMeansCluster(),
                  'Random' : randomCluster(),
                  'Target' : perfectCluster(target)}

    testclusterers(data, n_cluster, **clusterers)