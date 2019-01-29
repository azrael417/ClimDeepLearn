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

#arguments: 
#$1 - source dir
#$2 - destination dir
#$3 - train count
#$4 - validation count
#$5 - test count

#training and validation in one go
#convert mode "climate:" very slow due to poor hdf5 performance
cmd="python ./parallel_stagein.py --targets ${2}/train ${2}/validation ${2}/test --cvt "climate:" --workers 8 --seed 7919 --counts ${3} ${4} ${5} --mkdir ${1}/train ${1}/validation ${1}/test"
#echo ${cmd}
${cmd}
cp ${1}/stats.h5 ${2}/train/
cp ${1}/stats.h5 ${2}/validation/
cp ${1}/stats.h5 ${2}/test/
