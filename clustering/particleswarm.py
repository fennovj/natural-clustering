__author__ = 'Fenno'

from clustering import Clustering
from app.score import score, getlabels
from numpy.random import randint, rand
from numpy import zeros, shape, min, argmin, copy


class ParticleSwarmCluster(Clustering):

    def __init__(self, n_particles=10, n_iterations=1000, w=0.72, c1=1.49, c2=1.49, norm=2, printfreq=50):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.norm = norm
        self.printfreq = printfreq  # For no printing, give a value like 0.1 or float('inf')

    def cluster(self, data, n_clusters):

        n, d = shape(data)
        locations = zeros((self.n_particles, n_clusters, d))
        bestlocations = copy(locations)

        for i in range(self.n_particles):
            for j in range(n_clusters):
                locations[i, j, :] = copy(data[randint(n), :])  # Initialize cluster centers to random datapoints

        velocities = zeros((self.n_particles, n_clusters, d))

        bestscores = [score(data, centroids=locations[i, :, :], norm=self.norm) for i in range(self.n_particles)]
        sbestlocation = copy(locations[argmin(bestscores), :, :])
        sbestscore = min(bestscores)

        for i in range(self.n_iterations):
            if i % self.printfreq == 0:
                print "Iteration", i, "best score:", sbestscore
            for j in range(self.n_particles):
                r = rand(n_clusters, d)
                s = rand(n_clusters, d)
                velocities[j, :, :] = (self.w * velocities[j, :, :]) + \
                                      (self.c1 * r * (bestlocations[j, :, :] - locations[j, :, :])) + \
                                      (self.c2 * s * (sbestlocation - locations[j, :, :]))
                locations[j, :, :] = locations[j, :, :] + velocities[j, :, :]
                currentscore = score(data, centroids=locations[j, :, :], norm=self.norm)
                if currentscore < bestscores[j]:
                    bestscores[j] = currentscore
                    bestlocations[j, :, :] = locations[j, :, :]
                    if currentscore < sbestscore:
                        sbestscore = currentscore
                        sbestlocation = copy(locations[j, :, :])

        return getlabels(data, centroids=sbestlocation, norm=self.norm)
