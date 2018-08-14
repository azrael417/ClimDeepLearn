#!/bin/bash

# load modules
module load xl
module load cuda/9.2.88
module load essl

#load old spectrum
module load spectrum-mpi/10.2.0.4-20180716

#set core files to 0 size
ulimit -c 0

# Disable multiple threads
export OMPI_MCA_osc_pami_allow_thread_multiple=0

#enable adaptive routing
#export PAMI_IBV_ENABLE_OOO_AR=1
#export PAMI_IBV_QP_SERVICE_LEVEL=8

#disable adaptive routing
export PAMI_IBV_ENABLE_OOO_AR=0
export PAMI_IBV_QP_SERVICE_LEVEL=0

#python env
export VENV=pyvenv_summit_8.6.18
