#!/bin/bash

#load modules
#module unload python/2.7-anaconda-4.4
#module load python/3.6-anaconda-4.4 
#source activate helper-env

#some parameters
#datapath=/global/cscratch1/sd/tkurth/gb2018/tiramisu/segm_h5_v3_reformat
datapath=/project/projectdirs/dasrepo/gb2018/tiramisu/segm_h5_v3_reformat
#datapath=/data1/tkurth/tiramisu/segm_h5_v3_old
maskpath=/global/cscratch1/sd/amahesh/gb_output/test_run_nn256_np1536_j143026_convergence/output_test_3
outpath=/global/cscratch1/sd/mayur/gb2018/images_sean_contour_test_3

#do the plotting
python plot_masks.py --datapath=${datapath} --maskpath=${maskpath} --outpath=${outpath}
