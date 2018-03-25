#!/bin/bash

#arguments:
#$1 - source dir
#$2 - destination dir
#$3 - number of files
echo "calling parallel_stagein.py"
rm -f ${2}/*
python ./parallel_stagein.py --target ${2} --cvt "climate:fp32:0,1,2,10" --workers 8 --count ${3} --mkdir ${1}
cp ${1}/stats.h5 ${2}/

