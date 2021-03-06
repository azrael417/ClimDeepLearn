#!/bin/bash

#see if nvme is there
echo $(hostname) $(df -h ${2} | tail -n1)

#create directory if not exists and clean up otherwise
if [ -d "${2}" ]; then
    rm -rf ${2}/*
fi
if [ ! -d "${2}" ]; then
    mkdir -p ${2}
fi

#activate
source activate tensorflow-py27

#arguments: 
#$1 - source dir
#$2 - destination dir
#$3 - train count
#$4 - validation count

#training and validation in one go
python ./parallel_stagein.py --targets ${2}/train/data ${2}/validation/data --cvt "climate:" --workers 8 --seed 7919 --counts ${3} ${4} --mkdir ${1}/train ${1}/validation
cp ${1}/stats.h5 ${2}/train/data
cp ${1}/stats.h5 ${2}/validation/data
