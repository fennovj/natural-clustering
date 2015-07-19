__author__ = 'Fenno'


from numpy import shape, array, sort, copy, zeros, sum, nonzero
from numpy.random import permutation
from scipy.spatial.distance import pdist, squareform
from clustering import Clustering

from matplotlib.pyplot import matshow, show, plot
from app.score import score
from sklearn.datasets import load_iris


class CACluster(Clustering):

    def __init__(self, r=1.0, norm=2, maxiter=5, printfreq=float('inf'), tracktape=False):
        self.r = r  # either an integer, or a fraction of the dataset)
        self.norm = norm
        self.maxiter = maxiter
        self.printfreq = printfreq
        self.tapehistory = None
        self.tracktape = tracktape

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
        maxr = int(self.r * n) if self.r <= 1 else self.r

        distances = squareform(pdist(data, metric='minkowski', p=self.norm))
        tape = permutation(n)
        boundaries = self.getboundaries(tape, distances, n_clusters)
        changemade = 0
        if self.tracktape:
            self.tapehistory = zeros((self.maxiter * n, n), dtype='int')

        for count in range(self.maxiter):
            if count % self.printfreq == 0:
                print "CACluster at iteration", count, changemade, "changes made since previous print."
                changemade = 0

            for i in range(n):
                for ir in range(3, maxr):
                    dist1 = distances[tape[i], tape[(i+1) % n]]
                    dist2 = distances[tape[i], tape[(i+ir) % n]]
                    if dist2 < dist1:
                        t = copy(tape[(i+1) % n])
                        tape[(i+1) % n] = copy(tape[(i+ir) % n])
                        tape[(i+ir) % n] = t
                        changemade += 1

                if self.tracktape:
                    self.tapehistory[(count*n) + i, :] = copy(tape)

            newboundaries = self.getboundaries(tape, distances, n_clusters)
            if (boundaries == newboundaries).all():
                break
            boundaries = newboundaries

        return self.assignclasses(tape, distances, n_clusters)


def plotclusteringattempt(clusterer=None, r=1.0):
    if clusterer is None:
        clusterer = CACluster(r=r)
    clusterer.tracktape = True  # nececcary for plotting everything

    irisdata, iristarget, iris_n_cluster = load_iris().data, load_iris().target, 3
    labels = clusterer.cluster(irisdata, iris_n_cluster)
    dist = squareform(pdist(irisdata, metric='minkowski', p=clusterer.norm))
    endindex = nonzero(sum(clusterer.tapehistory, axis=1))[0][-1]
    lasttape = clusterer.tapehistory[endindex, :]
    m = len(iristarget)
    chain = array([dist[lasttape[j], lasttape[(j+1) % m]] for j in range(m)])
    print "chain", chain
    lastboundaries = clusterer.getboundaries(lasttape, dist, iris_n_cluster)

    clusterpicture = iristarget[clusterer.tapehistory]
    for b in lastboundaries:
        clusterpicture[:, b] = iris_n_cluster

    plot(chain)
    print "number of iterations", endindex
    print "final tape", lasttape
    print "final true classes", clusterpicture[endindex, :]
    print "score: ", score(irisdata, labels=labels, norm=clusterer.norm)

    matshow(clusterpicture[:endindex, :])
    show()

if __name__ == '__main__':
    plotclusteringattempt(r=0.65)
