# cython: profile=True
import numpy as np

def ravel_shift(   indices, \
                   arrayRank, \
                   arrayShape, \
                   dimension,  \
                   amount,     \
                   dimensionWraps):
    """Return the raveled index of a shifted version of indices, where a
    specific dimension has been shifted by a certain amount.  If wrapping is
    not flagged and the shift is out of bounds, returns -1"""

    runningProduct = 1
    i = 0

    #Loop over dimensions, starting at the rightmost dimension
    for n in range(arrayRank,0,-1):
        #Calculate the running product of dimension sizes
        if( n != arrayRank):
            runningProduct *= arrayShape[n]

        #Set the current index
        thisIndex = indices[n-1]

        np = n-1
        if(np == dimension):
            #If this is the shifting dimension,
            #increment it
            thisIndex += amount

            #Check if we need to deal with a
            #wrap around dimension
            if(dimensionWraps):
                if(thisIndex < 0):
                    thisIndex += arrayShape[np]
                if(thisIndex >= arrayShape[np]):
                    thisIndex -= arrayShape[np]

            #Check if the current index is out of bounds;
            #return -1 if so
            if(thisIndex < 0 or thisIndex >= arrayShape[np]):
                i = -1
                break

        #increment the counter
        i += runningProduct*thisIndex

    #Check whether the index is within the memory bounds of the array
    #return the -1 flag if not
    runningProduct *= arrayShape[0]
    if(i >= runningProduct or i < 0):
        i = -1

    return i

def findNeighbors(   raveledStartIndex, \
                     searchThreshold, \
                     arrayShape, \
                     arrayRank, \
                     dimensionWraps, \
                     inputArray, \
                     isNotSearched, \
                   ):
    """Does a flood fill algorithim on inputArray in the vicinity of
    raveledStartIndex to find contiguous areas where raveledStartIndex > searchThreshold 
    
        input:
        ------
            raveledStartIndex   :   (integer) the index of inputArray.ravel() at which to start

            searchThreshold :   The threshold for defining fill regions
                                (inputArray > searchThreshold)

        output:
        -------
            A list of N-d array indices.
    
    """

    
    #Initialize the contiguous index list
    contiguousIndices = []

    #Initialize the search list
    itemsToSearch = [raveledStartIndex]

    while itemsToSearch != []:

        #Get the index of the current item
        itemTuple = np.unravel_index(itemsToSearch[0],arrayShape)

        for r in range(arrayRank):
            #Shift the current coordinate to the right by 1 in the r dimension
            shiftAmount = 1
            testIndexRight = ravel_shift( \
                                        itemTuple, \
                                        arrayRank, \
                                        arrayShape, \
                                        r, \
                                        shiftAmount,
                                        dimensionWraps[r])

            #Check that this coordinate is still within bounds
            if(testIndexRight >= 0):
                #Check if this index satisfies the search condition
                if(inputArray[testIndexRight] > searchThreshold and \
                        isNotSearched[testIndexRight] == 1):
                    #Append it to the search list if so
                    itemsToSearch.append(testIndexRight)
                    #Flags that this cell has been searched
                    isNotSearched[testIndexRight] = 0


            #Shift the current coordinate to the right by 1 in the r dimension
            shiftAmount = -1
            testIndexLeft = ravel_shift( \
                                        itemTuple, \
                                        arrayRank, \
                                        arrayShape, \
                                        r, \
                                        shiftAmount,
                                        dimensionWraps[r])

            #Check that this coordinate is still within bounds
            if(testIndexLeft > 0):
                #Check if this index satisfies the search condition
                if(inputArray[testIndexLeft] > searchThreshold and \
                        isNotSearched[testIndexLeft] == 1 ):
                    #Append it to the search list if so
                    itemsToSearch.append(testIndexLeft)
                    #Flags that this cell has been searched
                    isNotSearched[testIndexLeft] = 0

 

        #Flag that this index has been searched
        #isNotSearched[tuple(itemsToSearch[0])] = 0
        #Now that the neighbors of the first item in the list have been tested,
        #remove it from the list and put it in the list of contiguous values
        contiguousIndices.append(itemsToSearch.pop(0))

    #Return the list of contiguous indices (converted to index tuples)
    return np.unravel_index(contiguousIndices,arrayShape)

def floodFillSearch( \
                inputArray, \
                searchThreshold = 0.0, \
                wrapDimensions = None):
    """Given an N-dimensional array, find contiguous areas of the array
    satisfiying a given condition and return a list of contiguous indices
    for each contiguous area.
        
        input:
        ------

            inputArray      :   (array-like) an array from which to search
                                contiguous areas

            searchThreshold :   The threshold for defining fill regions
                                (inputArray > searchThreshold)

            wrapDimensions :    A list of dimensions in which searching
                                should have a wraparound condition

        output:
        -------

            An unordered list, where each item corresponds to a unique
            contiguous area for which inputArray > searchThreshold, and
            where the contents of each item are a list of array indicies
            that access the elements of the array for a given contiguous
            area.

    """
    #Determine the rank of inputArray
    try:
        arrayShape = np.array(np.shape(inputArray))
        arrayRank = len(arrayShape)
        numArrayElements = np.prod(arrayShape)
    except BaseException as e:
        raise ValueError("inputArray does not appear to be array like.  Error was: {}".format(e))

    #Set the dimension wrapping array
    dimensionWraps = arrayRank*[False]
    if wrapDimensions is not None:
        try:
            for k in list(wrapDimensions):
                dimensionWraps[k] = True
        except BaseException as e:
            raise ValueError("wrapDimensions must be a list of valid dimensions for inputArray. Original error was: {}".format(e))

    #Set an array of the same size indicating which elements have been set
    isNotSearched = np.ones(arrayShape,dtype = 'int')

    #Set the raveled input array
    raveledInputArray = inputArray.ravel()
    #And ravel the search inidcator array
    raveledIsNotSearched = isNotSearched.ravel()
    
    #Set the search list to null
    contiguousAreas = []

    #Loop over the array
    for i in range(numArrayElements):
        #print "{}/{}".format(i,numArrayElements)
        #Check if the current element meets the search condition
        if raveledInputArray[i] > searchThreshold and raveledIsNotSearched[i]:
            #Flag that this cell has been searched
            raveledIsNotSearched[i] = 0

            #If it does, use a flood fill search to find the contiguous area surrouinding
            #the element for which the search condition is satisified. At very least, the index
            #of this element is appended to contiguousAreas
            contiguousAreas.append(\
                                    findNeighbors(  i,  \
                                                    searchThreshold,    \
                                                    arrayShape,         \
                                                    arrayRank,          \
                                                    dimensionWraps,     \
                                                    raveledInputArray,         \
                                                    raveledIsNotSearched      ))

        else:
            #Flag that this cell has been searched
            raveledIsNotSearched[i] = 0
                                    


    #Set the list of contiguous area indices
    return contiguousAreas


