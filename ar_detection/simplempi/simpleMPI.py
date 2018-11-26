
class simpleMPI:
    """A simple wrapper around mpi4py that offers simple scattering of a list of objects.

    This is useful for embarassingly parallel, SPMD, type tasks that simply need to work on a list of things.

    example usage:

        import simpleMPI

        #Initialize MPI
        smpi = simpleMPI.simpleMPI()

        #Make a list of things (20 numbers in this case)
        testList = range(20)

        #Scatter the list to all processors (myList differs among processes now)
        myList = smpi.scatterList(testList)

        #Print the list contents (as well as the rank of the printing process)
        smpi.pprint(myList)

"""


    def __init__(   self, \
                    useMPI = True):
        """A simple wrapper around mpi4py that offers simple scattering of a list of objects.

            input:
            ------

            useMPI      :   flag whether to use mpi4py (False is useful for
                            use/debugging in situations where mpi4py is unavailable)

        """

        #Save whether we are using MPI
        self.useMPI = useMPI

        #If MPI is being used, initialize it and save the number of processors/processor size
        if(useMPI):
          #Initialize MPI
          from mpi4py import MPI
          #Get the global communicator
          comm = MPI.COMM_WORLD
          #Get this processes's rank
          rank = comm.Get_rank()
          #Get the total number of processors
          mpisize = comm.Get_size()
        #If MPI isn't being used, simply set mpisize to 1 and set dummy values for comm and rank
        else:
          comm = 0
          rank = 0
          mpisize = 1

        #Save the MPI paramters to the class
        self.comm = comm
        self.rank = rank
        self.mpisize = mpisize

        return

    def doSyncBarrier(self):
        """Sets a synchronization barrier."""
        if self.useMPI:
            self.comm.Barrier()

        return

    def scatterList(self,inlist):
        """Scatter a list of objects to all participating processors."""
        if(self.useMPI):
            #If this is the root processor, divide the list as evenly as possible among processors
            # _divideListForScattering() returns a list of lists, with `mpisize` lists in the top level list
            if self.rank == 0:
                try:
                    scatterableList = self._divideListForScattering(inlist)
                except:
                    scatterableList = self._divideDictForScattering(inlist)
            else:
                scatterableList = None

            #Scatter the lists to the other processes
            myList = self.comm.scatter(scatterableList,root=0)
        else:
            #If we aren't using MPI, simply return the given list
            myList = inlist

        #Return this processor's list
        return myList

    def _divideListForScattering(self,inlist):
        """returns a list of lists, with `self.mpisize` lists in the top level list"""

        #Create a list that explicitly has the proper size
        outlist = [ [] for i in range(self.mpisize) ]

        n = 0
        #Go through each item in the input list and append it to the output list
        #Cycle through the indices of the output list so that they input list is
        #dividided as evenly as possible
        for i in range(len(inlist)):
            outlist[n].append(inlist[i])
            n = n + 1
            if(n >= self.mpisize):
                n = 0

        #Return the list
        return outlist

    def _divideDictForScattering(self,indict):
        """returns a list of dicts, with `self.mpisize` lists in the top level list"""

        #Create a list that explicitly has the proper size
        outlist = [ {} for i in range(self.mpisize) ]

        n = 0
        #Go through each item in the input list and append it to the output list
        #Cycle through the indices of the output list so that they input list is
        #dividided as evenly as possible
        for item in indict:
            outlist[n][item] = indict[item]
            n = n + 1
            if(n >= self.mpisize):
                n = 0

        #Return the list
        return outlist


    def pprint(self,message):
        """Does a parallel-friendly print (with information about the printing processor)"""
        print("(rank {}/{}): {}".format(self.rank+1,self.mpisize,message))


if __name__ == "__main__":

    #Initialize MPI
    smpi = simpleMPI()

    #Make a list of things (20 numbers in this case)
    testList = range(20)

    #Scatter the list to all processors (myList differs among processes now)
    myList = smpi.scatterList(testList)

    #Print the list contents (as well as the rank of the printing process)
    smpi.pprint(myList)


    #Test scattering a dict object
    testDict = {"a" : 1, "b" : 10, "c" : "a string"}

    #Scatter the dict
    myDict = smpi.scatterList(testDict)

    #Print the dict contents and the rank
    smpi.pprint(myDict)

