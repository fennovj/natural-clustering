# -*- coding: utf-8 -*-
"""
Created on Sun Jun 07 21:50:53 2015

@author: Fenno
"""

import numpy as np

def centroidScore(datapoint, clusters, pheromone):
    
    
    
    
    
    
def acocCluster(data, n_clusters, R = 10, q0 = 0.0001, n_iter = 1000, beta = 2, ro = 0.1):
    
    n_samples = np.size(data)[0]
    pheromone = 0.75 * np.ones((n_samples, n_clusters))