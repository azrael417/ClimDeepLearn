#!/bin/bash

#arguments: 
#$1 - source dir
#$2 - destination dir
#$3 - maximum files to stage (total, not per node). Set to -1 if you want to stage everything

python ./stagein.py --prefix="data" --input_path=${1} --output_path=${2} --max_files=${3} --clean
cp ${1}/stats.h5 ${2}/
