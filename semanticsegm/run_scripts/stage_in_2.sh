#!/bin/bash

#necessary stuff
module unload PrgEnv-cray
module load PrgEnv-gnu
module load gcc/5.3.0
module load cudatoolkit/8.0.61_2.4.3-6.0.4.0_3.1__gb475d12
source activate tensorflow-mpi4py

#arguments: 
#$1 - source dir
#$2 - destination dir

echo "Stage-in started"
python ../tiramisu-tf/stagein.py --prefix="data" --input_path=${1} --output_path=${2} --max_files 160
cp ${1}/stats.h5 ${2}/
echo "Stage-in done"

