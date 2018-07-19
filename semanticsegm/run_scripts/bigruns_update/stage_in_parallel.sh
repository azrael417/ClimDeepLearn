#!/bin/bash

#create directory if not exists
if [ ! -d "${2}" ]; then
    mkdir -p ${2}
fi

VENV=pyvenv_summit_7.5.18
source ${2}/../${VENV}/bin/activate

export OMPI_MCA_osc_pami_allow_thread_multiple=1

#arguments: 
#$1 - source dir
#$2 - destination dir
#$3 - count

python ./parallel_stagein.py --target=${2}/data --cvt "climate:" --workers 8 --seed 7919 --count ${3} --mkdir ${1}/data-*
cp ${1}/stats.h5 ${2}/data

