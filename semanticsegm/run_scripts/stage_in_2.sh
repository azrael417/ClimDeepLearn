#!/bin/bash

#arguments: 
#$1 - source dir
#$2 - destination dir

python ./stagein.py --prefix="data" --input_path=${1} --output_path=${2} --max_files 2400
cp ${1}/stats.h5 ${2}/

