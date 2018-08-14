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
    AP.add_argument("--output_path",type=str,help="Path to write output file to, needs to be visible from rank 0.")
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
        if parsed.max_files > 0:
            files = files[:parsed.max_files]

        #report what was found
        print("Summarizing {} files on {} ranks.".format(len(files), comm_size))

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

    #compute stats:
    count = 1
    with h5.File(os.path.join(parsed.input_path, files[0]),'r') as f:
        meanstats = f["climate"]["stats"][...]
        meanstats_labels = f["climate"]["labels_stats"][...].astype(np.int64)
        
    for filename in files[1:]:
        with h5.File(os.path.join(parsed.input_path,filename),'r') as f:
          
            #data stats
            stats = f["climate"]["stats"][...]
            # stats order is mean, max, min, stddev
            #keep sum for average
            meanstats[:,0] += stats[:,0]
            #maximum for max
            meanstats[:,1] = np.maximum(meanstats[:,1], stats[:,1])
            #minimum for min
            meanstats[:,2] = np.minimum(meanstats[:,2], stats[:,2])
            #TODO: check
            meanstats[:,3] += (np.square(stats[:,3]) + np.square(stats[:,0]))
            
            #labels stats just need to be summed
            meanstats_labels += f["climate"]["labels_stats"][...].astype(np.int64)
            
            #increase count
            count += 1
            
            
    #global reductions
    # count
    total_count = comm.allreduce(count, op=MPI.SUM)
    
    #data stats
    # average = sum/count
    sendbuff = meanstats[:,0].copy() / float(total_count)
    recvbuff = sendbuff.copy()
    comm.Allreduce(sendbuff, recvbuff, op=MPI.SUM)
    meanstats[:,0] = recvbuff[:]
    
    #max
    sendbuff = meanstats[:,1].copy()
    recvbuff = sendbuff.copy()
    comm.Allreduce(sendbuff, recvbuff, op=MPI.MAX)
    meanstats[:,1] = recvbuff[:]

    #min
    sendbuff = meanstats[:,2].copy()
    recvbuff = sendbuff.copy()
    comm.Allreduce(sendbuff, recvbuff, op=MPI.MIN)
    meanstats[:,2] = recvbuff[:]

    #stdev
    sendbuff = meanstats[:,3].copy()
    recvbuff = sendbuff.copy()
    comm.Allreduce(sendbuff, recvbuff, op=MPI.SUM)
    meanstats[:,3] = np.sqrt( (recvbuff[:] - np.square(meanstats[:,0])) / total_count )

    #labels stats
    # average = sum/count
    sendbuff = meanstats_labels.copy().astype(np.float32) / float(total_count)
    recvbuff = sendbuff.copy()
    comm.Allreduce(sendbuff, recvbuff, op=MPI.SUM)
    meanstats_labels = recvbuff.copy()

    #TODO: merge stddev properly
            
    #write the stuff to a file on each rank:
    if comm_rank == 0:
        outfilename = os.path.join(parsed.output_path, 'stats.h5')
        with h5.File(outfilename, 'w') as f:
            #create group
            f.create_group("climate")
            #create stats dataset
            dset_s = f.create_dataset("climate/stats", (16,4))
            dset_s[...] = meanstats[...]
            #create labelstats dataset
            dset_l = f.create_dataset("climate/labels_stats", (2,3))
            dset_l[...] = meanstats_labels[...]
            
if __name__ == "__main__":
    main()
