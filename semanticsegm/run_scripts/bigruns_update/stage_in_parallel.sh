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

#stage environment
cp /gpfs/alpinetds/world-shared/ven201/seant/climate/${VENV}.tar ${2}
tar xf ${2}/${VENV}.tar -C ${2}

#activate
source ${2}/${VENV}/bin/activate

#some MPI pars
export OMPI_MCA_osc_pami_allow_thread_multiple=1

#disable adaptive routing
export PAMI_IBV_ENABLE_OOO_AR=0
export PAMI_IBV_QP_SERVICE_LEVEL=0

#arguments: 
#$1 - source dir
#$2 - destination dir
#$3 - train count
#$4 - validation count

#training and validation in one go
python ./parallel_stagein.py --targets ${2}/train/data ${2}/validation/data --cvt "climate:" --workers 8 --seed 7919 --counts ${3} ${4} --mkdir ${1}/train ${1}/validation
cp ${1}/stats.h5 ${2}/train/data
cp ${1}/stats.h5 ${2}/validation/data
