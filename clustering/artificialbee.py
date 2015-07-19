__author__ = 'Fenno'

from numpy import zeros, shape, copy, argmin, array, nonzero
from numpy.random import randint, choice
from clustering import Clustering
from app.score import score, randfloat, getlabels


class ArtificialBeeCluster(Clustering):

    def __init__(self, n_bees=20, n_iter=100, limit=50, norm=2, printfreq=float('inf')):
        self.n_bees = n_bees
        self.n_iter = n_iter
        self.limit = limit
        self.norm = norm
        self.printfreq = printfreq

    @staticmethod
    def update(k, locations, newscore, newcentroids, currentscore, bestscore, bestlocation, changecount, change):
        if newscore < currentscore[k]:
            locations[k, :, :] = newcentroids
            currentscore[k] = newscore
            changecount[k] = 0
            if newscore < bestscore:
                bestlocation = newcentroids
                bestscore = newscore
        elif change:
            changecount[k] += 1
        return locations, currentscore, bestscore, bestlocation, changecount

    def getnewcentroids(self, data, locations, k):
        a = randint(self.n_bees)
        theta = randfloat(-1.0, 1.0)
        newcentroids = locations[k, :, :] + (theta * (locations[k, :, :] - locations[a, :, :]))
        newscore = score(data, centroids=newcentroids, norm=self.norm)
        return newcentroids, newscore

    def cluster(self, data, n_clusters):

        n, d = shape(data)
        locations = zeros((self.n_bees, n_clusters, d))

        for i in range(self.n_bees):
            for j in range(n_clusters):
                locations[i, j, :] = copy(data[randint(n), :])  # Initialize cluster centers to random datapoints

        currentscore = array([score(data, centroids=locations[i, :, :], norm=self.norm) for i in range(self.n_bees)])
        changecount = zeros(self.n_bees)
        bestlocation = copy(locations[argmin(currentscore), :, :])
        bestscore = min(currentscore)

        for it in range(self.n_iter):
            if it % self.printfreq == 0:
                print "Artificial Bee iteration", it, "best score:", bestscore

            for k in range(self.n_bees):
                newcentroids, newscore = self.getnewcentroids(data, locations, k)
                locations, currentscore, bestscore, bestlocation, changecount = self.update(
                    k, locations, newscore, newcentroids, currentscore, bestscore, bestlocation, changecount, True)

            for _ in range(self.n_bees):
                k = choice(self.n_bees, p=currentscore/sum(currentscore))
                newcentroids, newscore = self.getnewcentroids(data, locations, k)
                locations, currentscore, bestscore, bestlocation, changecount = self.update(
                    k, locations, newscore, newcentroids, currentscore, bestscore, bestlocation, changecount, False)

            for k in nonzero(changecount >= self.limit):
                newcentroids = array([data[randint(n), :] for _ in range(n_clusters)], copy=True)
                newscore = score(data, centroids=newcentroids, norm=self.norm)
                locations, currentscore, bestscore, bestlocation, changecount = self.update(
                    k, locations, newscore, newcentroids, currentscore, bestscore, bestlocation, changecount, False)

        return getlabels(data, centroids=bestlocation, norm=self.norm)
