# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 13:21:01 2015

@author: Fenno
"""

from numpy import array, shape
from numpy.random import rand
from clustering.clustering import RandomCluster, PerfectCluster
from clustering.kmeans import KMeansCluster
# from clustering.cellular_automata import CACluster
from clustering.particleswarm import ParticleSwarmCluster
from clustering.antcolony import AntColonyCluster
from clustering.artificialbee import ArtificialBeeCluster
from score import score


def simpledataset(n=20, spread=0.6):
    seeds = array([(0.5, 0.5), (0.5, 1.5), (1.5, 0.5), (1.5, 1.5)] * n)
    labels = array([0, 1, 2, 3] * n)
    offsets = spread * rand((4*n), 2) - (spread/2)
    return seeds+offsets, labels


def shuffle(data, classes=None):
    """ Shuffle the rows of data.	"""    
    from random import shuffle as rshuffle 
    key = array(range(shape(data)[0]))
    rshuffle(key)
    data = data[key, :]
    classes = classes[key]
    return data, classes


def testclusterers(data, n_cluster, **clusterlist):
    keys = clusterlist.keys()
    labels = [clusterlist[key].cluster(data, n_cluster) for key in keys]

    for i, key in enumerate(keys):
        print key, "score:", score(data, labels[i])

if __name__ == '__main__':
    # from sklearn import datasets
    # iris = datasets.load_iris().data
    # target = datasets.load_iris().target
    # iris, target = shuffle(iris, target)
    
    simpledata, target = simpledataset(20, 0.6)
    
    simple_n_cluster = 4  # 3 for iris, 4 for simple

    clusterers = {'kMeans': KMeansCluster(),
                  'Random': RandomCluster(),
                  # 'Cellular Automata': CACluster(),
                  'Particle Swarm': ParticleSwarmCluster(),
                  'Ant Colony': AntColonyCluster(),
                  'Artificial Bee': ArtificialBeeCluster(),
                  'Target': PerfectCluster(target)}

    testclusterers(simpledata, simple_n_cluster, **clusterers)
