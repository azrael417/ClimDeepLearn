import numpy as np
import h5py as h5
import os
import argparse


def main():
    
    AP = argparse.ArgumentParser()
    AP.add_argument("--input_files", type=str, nargs='+', required=True, help="Path to input files.")
    AP.add_argument("--output_file", type=str, required=True, help="Path to output file.")
    parsed = AP.parse_args()
    
    #compute stats:
    dstats=[]
    lstats=[]
    for filename in parsed.input_files:
      with h5.File(filename, 'r') as f:
        dstats.append(f["climate"]["stats"][...])
        lstats.append(f["climate"]["labels_stats"][...])

    #concatenate data
    dstats = np.stack(dstats, axis=-1)
    lstats = np.stack(lstats, axis=-1)

    #average the right way
    #labels are just averaged
    lstats = np.mean(lstats, axis=-1)
    
    #for dstats it is a bit more complicated:
    #average first entry
    dstats0 = np.mean(dstats[:,0,:], axis=-1)
    #max the second entry
    dstats1 = np.max(dstats[:,1,:], axis=-1)
    #min the third entry
    dstats2 = np.min(dstats[:,2,:], axis=-1)
    #this needs min subtraction before averaging:
    dstats3 = np.sqrt(np.mean(np.square(dstats[:,3,:]) + np.square(dstats[:, 0, :]), axis=-1) - np.square(dstats0))

    #merge back
    dstats = np.stack([dstats0, dstats1, dstats2, dstats3], axis=-1)

    #store it
    with h5.File(parsed.output_file, 'w-') as f:
        f["climate/stats"] = dstats[...]
        f["climate/labels_stats"] = lstats[...]

    
if __name__ == "__main__":
	main()
    
