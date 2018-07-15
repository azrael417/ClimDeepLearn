#!/bin/bash
# Copy and extract python environment (requires spectrum-mpi and cuda modules to be loaded)
cp /gpfs/alpinetds/world-shared/ven201/seant/climate/pyvenv_summit_v4.tar ${2}
tar xf ${2}/pyvenv_summit_v4.tar -C ${2}

VENV=pyvenv_summit_v4
source ${2}/${VENV}/bin/activate

export OMPI_MCA_osc_pami_allow_thread_multiple=1

#arguments: 
#$1 - source dir
#$2 - destination dir
#$3 - count

python ./parallel_stagein.py --target=${2}/data --cvt "climate:" --workers 8 --seed 7919 --count ${3} --mkdir ${1}/data-*
cp ${1}/stats.h5 ${2}/data

