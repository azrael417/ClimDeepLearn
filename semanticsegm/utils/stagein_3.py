from mpi4py import MPI
import h5py as h5
import numpy as np
import shutil as sh
import os
import argparse
import random as rnd
import time


def main():
    AP = argparse.ArgumentParser()
    AP.add_argument("--prefix",type=str,default="",help="Prefix to separate the files.")
    AP.add_argument("--input_path",type=str,help="Path to read from. Needs to be visible by all the nodes.")
    AP.add_argument("--output_path",type=str,help="Path to write to, needs to be visible from the individual nodes writing the shard.")
    AP.add_argument("--max_files",type=int,help="Maximum number of files to copy.")
    parsed = AP.parse_args()

    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    #on rank zero, see how many files are there
    files_new = None
    if comm_rank == 0:            
        start = time.time()
        #check the files
        files = sorted([x for x in os.listdir(parsed.input_path) if x.startswith(parsed.prefix)])
        files = files[:parsed.max_files]

        num_files = len(files)
        print("Preparing {} files for staging.".format(num_files))

        #shuffle the list
        rnd.seed(1234)
        rnd.shuffle(files)

        for i,name in enumerate(files):
            print("Processing file {}: {}...".format(i+1,name))
            path = parsed.input_path + '/' + name

            # rank 0 reads file from gpfs
            f = h5.File(path, 'r')
            data = np.array(f['climate']['data'])
            labels = np.array(f['climate']['labels'])
            stats = np.array(f['climate']['stats'])
            f.close()

            # rank 0 broadcasts data
            name = comm.bcast(name, root=0)
            data = comm.bcast(data, root=0)
            labels = comm.bcast(labels, root=0)
            stats = comm.bcast(stats, root=0)

            # rank 0 saves new h5 file to nvme
            if not os.path.isdir(parsed.output_path):
              os.makedirs(parsed.output_path)

            outpath = parsed.output_path + '/' + name
            f = h5.File(outpath, 'w')
            g = f.create_group('climate')
            g.create_dataset('data', data = data)
            g.create_dataset('labels', data = labels)
            g.create_dataset('stats', data = stats)
            f.close()

    else:
        # other ranks receive data
        for i in xrange(parsed.max_files):
          name = None
          data = None
          labels = None
          stats = None
          name = comm.bcast(name, root=0)
          data = comm.bcast(data, root=0)
          labels = comm.bcast(labels, root=0)
          stats = comm.bcast(stats, root=0)

          # save new h5 file to nvme
          if not os.path.isdir(parsed.output_path):
            os.makedirs(parsed.output_path)

          outpath = parsed.output_path + '/' + name
          f = h5.File(outpath, 'w')
          g = f.create_group('climate')
          g.create_dataset('data', data = data)
          g.create_dataset('labels', data = labels)
          g.create_dataset('stats', data = stats)
          f.close()

    if comm_rank == 0:
        end = time.time()
        print("Staging done. Time: {} s".format(end-start))

if __name__ == "__main__":
    main()
