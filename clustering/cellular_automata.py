__author__ = 'Fenno'


from numpy import shape, array, sort, copy, zeros
from numpy.random import permutation
from scipy.spatial.distance import pdist, squareform
from clustering import Clustering


class CACluster(Clustering):

    def __init__(self, r=None, norm=2, maxiter=5000, printfreq=float('inf')):
        self.r = r
        self.norm = norm
        self.maxiter = maxiter
        self.printfreq = printfreq

    @staticmethod
    def getboundaries(tape, distances, n_clusters):
        n = len(tape)
        chainmap = array([distances[tape[i], tape[(i+1) % n]] for i in range(n)])
        return sort(chainmap.argsort()[-n_clusters:])

    @staticmethod
    def assignclasses(tape, distances, n_clusters):
        boundaries = CACluster.getboundaries(tape, distances, n_clusters) + 1
        result = zeros(len(tape), dtype='int32')
        for i in range(n_clusters - 1):
            result[boundaries[i % n_clusters]: boundaries[(i+1) % n_clusters]] = i
        result[boundaries[-1]:] = n_clusters-1
        result[:boundaries[0]] = n_clusters-1
        return result[tape.argsort()]

    def cluster(self, data, n_clusters):
        n, _ = shape(data)
        maxr = n / 2 if self.r is None else self.r

        distances = squareform(pdist(data, metric='minkowski', p=self.norm))
        tape = permutation(n)
        boundaries = self.getboundaries(tape, distances, n_clusters)
        changemade = 0

        for count in range(self.maxiter):
            if count % self.printfreq == 0:
                print "CACluster at iteration", count, changemade, "changes made this iteration."
            changemade = 0

            for i in range(n):
                for ir in range(3, maxr):
                    dist1 = distances[tape[i], tape[(i+1) % n]]  # minkowski(data[i,:], data[(i+1)%n,:],2)
                    dist2 = distances[tape[i], tape[(i+ir) % n]]  # minkowski(data[i,:], data[(i+r)%n,:],2)
                    if dist2 < dist1:
                        t = copy(tape[(i+1) % n])
                        tape[(i+1) % n] = copy(tape[(i+ir) % n])
                        tape[(i+ir) % n] = t
                        changemade += 1

            newboundaries = self.getboundaries(tape, distances, n_clusters)
            if (boundaries == newboundaries).all():
                break
            boundaries = newboundaries

        return self.assignclasses(tape, distances, n_clusters)
