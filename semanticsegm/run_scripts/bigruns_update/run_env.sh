#!/bin/bash

# load modules
module load xl
module load cuda/9.2.88
module load essl

#set core files to 0 size
ulimit -c 0

# Disable multiple threads
export OMPI_MCA_osc_pami_allow_thread_multiple=0

#disable adaptive routing
export PAMI_IBV_ENABLE_OOO_AR=0
export PAMI_IBV_QP_SERVICE_LEVEL=0
