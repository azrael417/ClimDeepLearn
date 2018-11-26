import pstats, cProfile

#import pyximport
#pyximport.install()

import cFloodFillSearch as floodFillSearch

from numpy import * 

def runTest():
    N = 2**9

    random.seed(0)
    testArray = random.normal(size =[N,N])
    #print(shape(testArray))

    cProfile.runctx("floodFillSearch.floodFillSearch(testArray)", globals(), locals(), "Profile.prof")
    #cProfile.runctx("floodFillSearch.floodFillSearch(testArray)", globals(), locals(), "Profile.prof")

    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()


    #############
    #############
    #############
    floodInds = floodFillSearch.floodFillSearch(testArray)

    areaSizes = array([ len(inds) for (inds,_) in floodInds])
    sortInds = argsort(areaSizes)

    indexArray = ma.zeros(shape(testArray))
    indexArray[:] = ma.masked

    for i in range(len(sortInds)):
        indexArray[floodInds[i]] = log(areaSizes[i])

    print(areaSizes[sortInds])

    import pylab as P
    P.imshow(indexArray,interpolation='nearest')
    P.show()


