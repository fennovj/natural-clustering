# -*- coding: utf-8 -*-
"""
Created on Mon Jun 01 19:42:16 2015

@author: Fenno
"""

from clustering import Clustering
from sklearn.cluster import KMeans as sklearnKMeans


class KMeansCluster(Clustering):

    def __init__(self, max_iter=300, n_init=12, n_jobs=1, tol=1e-4, verbose=0):
        self.max_iter = max_iter
        self.n_init = n_init
        self.n_jobs = n_jobs
        self.tol = tol
        self.verbose = verbose

    def cluster(self, data, n_clusters):
        model = sklearnKMeans(n_clusters=n_clusters, max_iter=self.max_iter,
                              n_init=self.n_init, n_jobs=self.n_jobs, tol=self.tol, verbose=self.verbose)
        return model.fit_predict(data)
