import numpy as np
import h5py as h5
import os
import sys
import argparse

#enable mpi4py
from mpi4py import MPI

def main():
    
    AP = argparse.ArgumentParser()
    AP.add_argument("--input_path", type=str, required=True, help="Path to input files.")
    AP.add_argument("--output_path", type=str, required=True, help="Path to output files.")
    AP.add_argument("--update", action='store_true', help="Check which files are already present in output_path and only generate the new ones.")
    parsed = AP.parse_args()
    
    #duplicate comm:
    comm = MPI.COMM_WORLD.Dup()
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    if comm_rank == 0:
        print("Copying data from {} to {}.".format(parsed.input_path,parsed.output_path))
    
    #create output path if not exists
    if comm_rank == 0 and not os.path.isdir(parsed.output_path):
        os.makedirs(parsed.output_path)
    
    #look for labels and data files
    infiles = sorted([x for x in os.listdir(parsed.input_path) if x.startswith("data")])
    
    #remove files which are already done
    if parsed.update:
        donefiles = set(os.listdir(parsed.output_path))
        infiles = [x for x in infiles if x not in donefiles]

    #print result
    if comm_rank == 0:
        print("Found {nf} files.".format(nf=len(infiles)))

    #chunk list of files
    chunk_size = len(infiles) // comm_size
    datafiles = infiles[ comm_rank*chunk_size : ((comm_rank+1)*chunk_size) ]
    
    #add the remainder to rank0:
    if comm_rank == 0 and (comm_size*chunk_size) < len(infiles):
        datafiles += [ infiles[ (comm_size)*chunk_size : ] ]

    #print rank and numfiles
    print("Rank {nr} will work on {nf} files.".format(nr=comm_rank, nf=len(datafiles)))
    comm.Barrier()
    
    #iterate over files and write
    for datafile in datafiles:

        #read data
        try:
            #data
            with h5.File(os.path.join(parsed.input_path, datafile),"r") as f:
                data = np.transpose(f["climate"]["data"][...],[2,0,1]).astype(np.float32)
                labels = np.expand_dims(f["climate"]["labels_0"][...].astype(np.int32), axis=0)
                labels = np.concatenate([labels, np.expand_dims(f["climate"]["labels_1"][...].astype(np.int32), axis=0)], axis=0)
                data_stats = np.transpose(f["climate"]["data_stats"][...],[1,0]).astype(np.float32)
            
            #compute labels stats
            labels_stats=np.zeros((2,3), dtype=np.int32)
            for l in range(0,2):
                uniqs, cnts = np.unique(labels[l,:,:], return_counts=True)
                uniqs = list(uniqs)
                cnts = list(cnts)
                cntdct = {x[0]:x[1] for x in zip(uniqs, cnts)}
                for item in cntdct:
                    labels_stats[l,item] = cntdct[item]

        except:
            print("cannot open {} for reading".format(datafile))
            continue
        
        #create output files
        outfilename = os.path.join(parsed.output_path, datafile)

        #write data
        try:
            #data
            with h5.File(outfilename,'w') as f:
                #create group
                f.create_group("climate")
                #create data dataset
                dset_d = f.create_dataset("climate/data", (16,768,1152))
                dset_d[...] = data[...]
                #create labels dataset
                dset_l = f.create_dataset("climate/labels", (2,768,1152))
                dset_l[...] = labels[...]
                #create data stats dataset
                dset_ds = f.create_dataset("climate/stats", (16,4))
                dset_ds[...] = data_stats[...]
                #create labels stats dataset
                dset_ls = f.create_dataset("climate/labels_stats", (2,3))
                dset_ls[...] = labels_stats[...]
                
        except:
            print("cannot open {} for writing".format(outfilename))
            continue


if __name__ == "__main__":
	main()
