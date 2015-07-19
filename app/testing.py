# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 13:21:01 2015

@author: Fenno
"""

from numpy import array, unique
from numpy.random import rand, permutation
from clustering.clustering import RandomCluster, PerfectCluster
from clustering.kmeans import KMeansCluster
from clustering.cellular_automata import CACluster
from clustering.particleswarm import ParticleSwarmCluster
from clustering.antcolony import AntColonyCluster
from clustering.artificialbee import ArtificialBeeCluster
from score import score
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


def testclusterers(data, n_cluster, **clusterlist):
    keys = clusterlist.keys()
    labels = [clusterlist[key].cluster(data, n_cluster) for key in keys]

    for i, key in enumerate(keys):
        print key, "score:", score(data, labels[i])

if __name__ == '__main__':
    # simpledata, simpletarget, simple_n_cluster = simpledataset()
    irisdata, iristarget, iris_n_cluster = realdataset()

    clusterers = {'kMeans': KMeansCluster(),
                  'Random': RandomCluster(),
                  'Cellular Automata': CACluster(printfreq=100),
                  'Particle Swarm': ParticleSwarmCluster(),
                  'Ant Colony': AntColonyCluster(),
                  'Artificial Bee': ArtificialBeeCluster(),
                  'Target': PerfectCluster(iristarget)}

    # testclusterers(simpledata, simple_n_cluster, **clusterers)
    testclusterers(irisdata, iris_n_cluster, **clusterers)
