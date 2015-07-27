# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 13:21:01 2015

@author: Fenno
"""

from clustering.clustering import RandomCluster, PerfectCluster
from clustering.kmeans import KMeansCluster
from clustering.cellular_automata import CACluster
from clustering.particleswarm import ParticleSwarmCluster
from clustering.antcolony import AntColonyCluster
from clustering.artificialbee import ArtificialBeeCluster
from score import score
from datasets import realdataset


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
                  'Cellular Automata': CACluster(r=0.6, printfreq=100),
                  'Particle Swarm': ParticleSwarmCluster(),
                  'Ant Colony': AntColonyCluster(),
                  'Artificial Bee': ArtificialBeeCluster(),
                  'Target': PerfectCluster(iristarget)}

    # testclusterers(simpledata, simple_n_cluster, **clusterers)
    testclusterers(irisdata, iris_n_cluster, **clusterers)
