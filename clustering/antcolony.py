__author__ = 'Fenno'

from numpy import shape, ones, zeros, array, argmax, average, copy
from numpy.random import permutation, randint, choice
from clustering import Clustering
from app.score import score, getlabels, randfloat
from scipy.spatial.distance import minkowski


class AntColonyCluster(Clustering):

    def __init__(self, n_ants=10, q0=0.0001, n_iter=100, beta=2, ro=0.1, t0=0.75, printfreq=float('inf')):
        self.n_ants = n_ants
        self.q0 = q0  # probability of exploiting
        self.n_iter = n_iter
        self.beta = beta  # metric for choosing best cluster for given datapoint
        self.ro = ro  # decay of existing pheromone (the lower, the less decay)
        self.t0 = t0  # initial pheromone
        self.printfreq = printfreq

    @staticmethod
    def centroidscore(datapoint, centroids, pheromone, beta):
        distances = array([minkowski(datapoint, centroids[i, :], beta) for i in range(shape(centroids)[0])])
        return pheromone * (1.0 / distances)

    def cluster(self, data, n_clusters):

        n_samples, _ = shape(data)
        assert self.n_ants < n_samples, "number of ants must be lower than number of samples"

        bestscore = float('inf')
        bestcentroids = None
        bestweights = None

        pheromone = self.t0 * ones((n_samples, n_clusters))

        for it in range(self.n_iter):

            for _ in range(self.n_ants):

                # memory = -1 * ones(n_samples, dtype='int')
                weights = zeros((n_samples, n_clusters), dtype='bool')
                centroids = array([data[randint(n_samples), :] for _ in range(n_clusters)], copy=True)

                for i in permutation(n_samples):
                    scores = self.centroidscore(data[i, :], centroids, pheromone[i, :], self.beta)

                    if randfloat() < self.q0:
                        j = argmax(scores)  # exploit
                    else:
                        j = choice(n_clusters, p=(scores / sum(scores)))  # explore

                    weights[i, j] = True
                    centroids[j, :] = average(data, axis=0, weights=weights[:, j])

                currentscore = score(data, centroids=centroids, norm=self.beta)
                if currentscore < bestscore:
                    bestscore = currentscore
                    bestcentroids = copy(centroids)
                    bestweights = copy(weights)

            pheromone = (self.ro * pheromone) + ((1.0 / bestscore) * bestweights)

            if it % self.printfreq == 0:
                print "Ant Colony iteration", it, "best score:", bestscore

        return getlabels(data, centroids=bestcentroids, norm=self.beta)
