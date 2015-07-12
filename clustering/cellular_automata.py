__author__ = 'Fenno'


from numpy import array, shape, sort, zeros, copy
from scipy.spatial.distance import minkowski
from clustering import Clustering


def swap(data, i, j):
    t = copy(data[i, :])
    data[i, :] = data[j, :]
    data[j, :] = t


def getboundaries(data, distmat, tape, n_clusters):
    n = shape(data)[0]
    distances = array([distmat[tape[i], tape[(i+1) % n]] for i in range(n)])
    return sort(distances.argsort()[-n_clusters:][::-1])


def findboundaries(data, distmat, tape, n_clusters):
    n = shape(data)[0]
    # distances = array([minkowski(data[tape[i],:], data[tape[(i+1)%n],:],norm) for i in range(n)])
    boundaries = getboundaries(data, distmat, tape, n_clusters)

    result = zeros(n, dtype='int32')
    for i in range(n_clusters - 1):
        result[boundaries[i % n_clusters]: boundaries[(i+1) % n_clusters]] = i
    result[boundaries[-1]:] = n_clusters-1
    result[:boundaries[0]] = n_clusters-1
    return result[tape]


class CACluster(Clustering):

    def __init__(self, r=None, norm=2, maxiter=2000):
        self.r = r
        self.norm = norm
        self.maxiter = maxiter

    def cluster(self, data, n_clusters):

        n = shape(data)[0]
        distances = array([[minkowski(data[i, :], data[j, :], self.norm) for i in range(n)] for j in range(n)])

        tape = range(n)
        rbackup = False

        if self.r is None:
            rbackup = True
            self.r = shape(data)[0] / 2
        changemade = count = 1
        equivalent = False
        boundaries = getboundaries(data, distances, tape, n_clusters)

        while (not equivalent) and changemade and count < self.maxiter:
            count += 1
            if count % 100 == 0:
                print count, changemade
            changemade = 0

            for ir in range(3, self.r):
                for i in range(n):
                    dist1 = distances[tape[i], tape[(i+1) % n]]  # minkowski(data[i,:], data[(i+1)%n,:],2)
                    dist2 = distances[tape[i], tape[(i+ir) % n]]  # minkowski(data[i,:], data[(i+r)%n,:],2)
                    if dist2 < dist1:
                        changemade += 1
                        t = tape[(i+1) % n]
                        tape[(i+1) % n] = tape[(i+ir) % n]
                        tape[(i+ir) % n] = t
                        # swap(data,(i+1)%n,(i+r)%n)
            newboundaries = getboundaries(data, distances, tape, n_clusters)
            equivalent = (boundaries == newboundaries).all()
            boundaries = newboundaries
        if rbackup:
            self.r = None
        return findboundaries(data, distances, tape, n_clusters)
