#!/bin/bash

#load modules
module load python/3.6-anaconda-4.4 

#some parameters
datapath=/global/cscratch1/sd/tkurth/gb2018/tiramisu/segm_h5_v3_reformat
maskpath=/global/cscratch1/sd/amahesh/gb_output/test_run_nn256_np1536_j143026_convergence/output_test

#run the plotting program
for maskfile in $(ls ${maskpath}/*.npz); do
    python plot_masks.py --datapath=${datapath} --masks=${maskfile}
    break
done

