#!/bin/bash
VENV=pyvenv_TF_spectrum
source ~/${VENV}/bin/activate

# Modify LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/sw/summit/cuda/9.1.85/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=~/Summit/nccl_2.2.10/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=~/hdf5-1.10.1/lib:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=~/${VENV}/lib/python2.7/site-packages/horovod/tensorflow:$LD_LIBRARY_PATH

export OMPI_MCA_osc_pami_allow_thread_multiple=1

#arguments: 
#$1 - source dir
#$2 - destination dir
#echo "calling parallel_stagein.py"
python ./parallel_stagein.py --target=${2} --cvt "climate:" --workers 8 --count ${3} --mkdir ${1}/data-*
cp ${1}/stats.h5 ${2}/

