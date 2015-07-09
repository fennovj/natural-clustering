__author__ = 'Fenno'


from numpy import array, shape, sort, zeros, copy
from scipy.spatial.distance import minkowski

def swap(data, i, j):
    t = copy(data[i,:])
    data[i,:] = data[j,:]
    data[j,:] = t

def getBoundaries(data, distmat, tape, n_clusters):
    n = shape(data)[0]
    distances = array([distmat[tape[i], tape[(i+1)%n]] for i in range(n)])
    return sort(distances.argsort()[-n_clusters:][::-1])

def findBoundaries(data, distmat, tape, n_clusters):
    n = shape(data)[0]
    #distances = array([minkowski(data[tape[i],:], data[tape[(i+1)%n],:],norm) for i in range(n)])
    boundaries = getBoundaries(data, distmat, tape, n_clusters)

    result = zeros(n,dtype='int32')
    for i in range(n_clusters -1):
        result[boundaries[i%n_clusters] : boundaries[(i+1)%n_clusters]] = i
    result[boundaries[-1]:]=n_clusters-1
    result[:boundaries[0]] =n_clusters-1
    return result[tape]

def CACluster(data, n_clusters, R = None, norm = 2, maxIter = 2000):

    n = shape(data)[0]
    distances = array([[minkowski(data[i,:], data[j,:], norm) for i in range(n) ] for j in range(n)])

    tape = range(n)

    if R is None:
        R = shape(data)[0] / 2
    changeMade = count = 1
    equivalent = False
    boundaries = getBoundaries(data,distances,tape,n_clusters)

    while((not equivalent) and changeMade and count < maxIter):
        count = count + 1
        if count % 100 == 0:
            print count, changeMade
        changeMade= 0

        for r in range(3, R):
            for i in range(n):
                dist1 = distances[tape[i], tape[(i+1)%n]]#minkowski(data[i,:], data[(i+1)%n,:],2)
                dist2 = distances[tape[i], tape[(i+r)%n]]#minkowski(data[i,:], data[(i+r)%n,:],2)
                if dist2 < dist1:
                    changeMade = changeMade + 1
                    t = tape[(i+1)%n]
                    tape[(i+1)%n] = tape[(i+r)%n]
                    tape[(i+r)%n] = t
                    #swap(data,(i+1)%n,(i+r)%n)
        newboundaries = getBoundaries(data,distances,tape,n_clusters)
        equivalent = (boundaries == newboundaries).all()
        boundaries = newboundaries
    return findBoundaries(data, distances, tape, n_clusters)
