__author__ = 'Fenno'

from numpy import shape
from matplotlib.pyplot import scatter, show


def twodplot(data, n_clusters, clusterer):
    assert shape(data)[1] == 2, "data must be twodimensional"
    labels = clusterer.cluster(data, n_clusters)
    scatter(data[:, 0], data[:, 1], c=labels, s=30)
    show()

if __name__ == '__main__':
    from clustering.kmeans import KMeansCluster
    from testing import simpledataset
    irisdata, _, n_cluster = simpledataset()
    twodplot(irisdata, n_cluster, KMeansCluster())