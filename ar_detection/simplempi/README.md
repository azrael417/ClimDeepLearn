# README #

A simple wrapper around mpi4py that offers simple scattering of a list (or dict) of objects.

This is useful for embarassingly parallel, SPMD, type tasks that simply need to work on a list of things.

example usage:

```python
import simpleMPI

#Initialize MPI 
smpi = simpleMPI.simpleMPI()

#Make a list of things (20 numbers in this case)
testList = range(20)

#Scatter the list to all processors (myList differs among processes now)
myList = smpi.scatterList(testList)

#Print the list contents (as well as the rank of the printing process)
smpi.pprint(myList)
```

Running this with mpirun on 6 processors shows that the list of 20 numbers gets
scattered as evenly as possible across all 6 processors:

```bash
$ mpirun -n 6 python simpleMPI.py 
(rank 1/6): [0, 6, 12, 18]
(rank 2/6): [1, 7, 13, 19]
(rank 4/6): [3, 9, 15]
(rank 6/6): [5, 11, 17]
(rank 5/6): [4, 10, 16]
(rank 3/6): [2, 8, 14]

```