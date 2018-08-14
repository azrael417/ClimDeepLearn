import numpy as np
import h5py as h5
import os
import argparse as AP


def main():
    
    AP = argparse.ArgumentParser()
    AP.add_argument("--input_files", type=str, narg='+', required=True, help="Path to input files.")
    AP.add_argument("--output_file", type=str, required=True, help="Path to output file.")
    parsed = AP.parse_args()
    
    #compute stats:
    for filename in parsed.input_files:
      with h5.File(os.path.join(filename, files[0]), 'r') as f:
        dstats = f["climate"]["stats"][...]
        lstats = f["climate"]["labels_stats"][...]
        
        print(dstat,lstats)
    
    
if __name__ == "__main__":
	main()
    