import numpy as np
import h5py as h5
import os

#path
input_path="/gpfs/alpinetds/scratch/tkurth/csc190/segm_h5_v3"
output_path="/gpfs/alpinetds/scratch/tkurth/csc190/segm_h5_v3_uncompressed"

#create output path if not exists
if not os.path.isdir(output_path):
    os.makedirs(output_path)

#look for labels and data files
labelfiles = sorted([x for x in os.listdir(input_path) if x.startswith("label")])
datafiles = sorted([x for x in os.listdir(input_path) if x.startswith("data")])

#only use the data where we have labels for and vice versa
datafiles = [x for x in datafiles if x.replace("data","labels") in labelfiles]
labelfiles = [x for x in labelfiles if x.replace("labels","data") in datafiles]

#parameters of merge
mergesize = 72
mergecount = 10

#do some cutoff
datafiles = datafiles[0:mergesize*mergecount]
labelfiles = labelfiles[0:mergesize*mergecount]

#iterate through and merge
data=np.zeros((mergesize,16,768,1152),dtype=np.float32)
data_stats=np.zeros((mergesize,16,4),dtype=np.float32)
labels=np.zeros((mergesize,768,1152),dtype=np.int32)
for mc in range(mergecount):
    for ms in range(mergesize):
        #data
        with h5.File(input_path+"/"+datafiles[ms+mergesize*mc],"r") as f:
            data[ms,:,:,:]=np.transpose(f["climate"]["data"][...],[2,0,1])
            data_stats[ms,:,:]=np.transpose(f["climate"]["data_stats"][...],[1,0])
        #label
        with h5.File(input_path+"/"+labelfiles[ms+mergesize*mc],"r") as f:
            labels[ms,:,:]=f["climate"]["labels"][...].astype(np.int32)
  
    #data
    with h5.File(output_path+'/data-chunk-'+str(mc)+'.h5','w',libver="latest") as f:
        #create group
        f.create_group("climate")
        #create data dataset
        dset_d = f.create_dataset("climate/data", (mergesize,16,768,1152), chunks=(1,16,768,1152))
        dset_d[...] = data[...]
        #create labels dataset
        dset_l = f.create_dataset("climate/labels", (mergesize,768,1152), chunks=(1,768,1152))
        dset_l[...] = labels[...]
        #create stats dataset
        dset_s = f.create_dataset("climate/stats", (mergesize,16,4), chunks=(1,16,4))
        dset_s[...] = data_stats[...]
