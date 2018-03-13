from mpi4py import MPI
import h5py as h5
import numpy as np
from shutil import copyfile
import os
import argparse
import random as rnd


def main():
    AP = argparse.ArgumentParser()
    AP.add_argument("--prefix",type=str,default="",help="Prefix to separate the files.")
    AP.add_argument("--input_path",type=str,help="Path to read from. Needs to be visible by all the nodes.")
    AP.add_argument("--output_path",type=str,help="Path to write to, needs to be visible from the individual nodes writing the shard.")
    AP.add_argument("--max_files",type=int,default=-1,help="Maximum number of files to copy.")
    parsed = AP.parse_args()

    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    #on rank zero, see how many files are there
    files_new = None
    if comm_rank == 0:
        #create output path if not exists
        if not os.path.isdir(parsed.output_path):
            os.makedirs(parsed.output_path)
            
        #check the files
        files = sorted([x for x in os.listdir(parsed.input_path) if x.startswith(parsed.prefix)])
        files = files[:parsed.max_files]

        #shuffle the list
        rnd.seed(1234)
        rnd.shuffle(files)

        #compute length and determine the set for every node:
        chunksize = len(files) // comm_size

        #combine the strings:
        files_new = []
        for i in range(comm_size):
            files_new.append(";".join(files[i*chunksize:(i+1)*chunksize]))
        
        #check max string length
        maxlen=0
        for string in files_new:
            maxlen = np.max([maxlen,len(string)])

        #pad the strings
        files_new = [x.ljust(maxlen,' ') for x in files_new]
        
    #comm the strings
    local_files = comm.scatter(files_new, 0)

    #split the strings
    files = local_files.strip().split(";")

    #do the stagein process
    for filename in files:
        infilename = parsed.input_path+'/'+filename
        outfilename = parsed.output_path+'/'+filename
        copyfile(infilename, outfilename)

if __name__ == "__main__":
    main()
