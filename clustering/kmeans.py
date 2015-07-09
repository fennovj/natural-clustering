# -*- coding: utf-8 -*-
"""
Created on Mon Jun 01 19:42:16 2015

@author: Fenno
"""

from sklearn.cluster import KMeans as sklearnKMeans

def kMeansCluster(data, n_clusters, max_iter = 300, n_init = 12, n_jobs = 1, tol = 1e-4, verbose = 0):    
    """Trans a KMeans model, using sklearn
    This method is nothing more than a wrapper for KMeans(args).fit_predict(data)
    """    
    model = sklearnKMeans(n_clusters = n_clusters, max_iter = max_iter, n_init = n_init, n_jobs = n_jobs, tol = tol, verbose = verbose)
    return model.fit_predict(data)


