#!/bin/bash

#arguments: 
#$1 - source dir
#$2 - destination dir

echo "Stage-in started"
python ../tiramisu-tf/stagein.py --prefix="data" --input_path=${1} --output_path=${2} --max_files 160
cp ${1}/stats.h5 ${2}/
echo "Stage-in done"

