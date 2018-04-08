#!/bin/bash

#arguments:
#$1 - source dir
#$2 - destination dir
#$3 - number of files
#echo "calling parallel_stagein.py at time $(date +%s)"
rm -f ${2}/*
python ./parallel_stagein.py --target ${2} --cvt "climate:fp32:0,1,2,10" --workers 8 --count ${3} --mkdir "${1}/data-*"
cp ${1}/stats.h5 ${2}/
#echo "finishing parallel_stagein.py at time $(date +%s)"
