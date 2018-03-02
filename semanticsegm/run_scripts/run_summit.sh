#!/bin/bash
#BSUB -nnodes 4
#BSUB -W 120 
#BSUB -P CSC275PRABHAT
#BSUB -alloc_flags "smt4 nvme"
#BSUB -J climseg_training
#BSUB -o out.%J
#BSUB -e out.%J

#set up modules
module unload spectrum-mpi
module use /sw/summit/diags/modulefiles
module load cuda
module load openmpi
source activate tensorflow

#set up library paths
export LD_LIBRARY_PATH=/sw/summit/diags/openmpi-3.0.0-nocuda/lib:/sw/summit/cuda/9.1.85/lib64:${HOME}/Anaconda/nvidia-stuff/nccl_2.2.10a0-1+cuda9.0_ppc64le/lib:${HOME}/anaconda3/envs/tensorflow/lib:${HOME}/anaconda3/envs/tensorflow/lib/python2.7/site-packages/horovod/tensorflow:${LD_LIBRARY_PATH}

#script in place
mkdir -p ${SWORK}/run/
cp stage_in.sh ${SWORK}/run/
cp ../tiramisu-tf/mascarpone-tiramisu-tf.py ${SWORK}/run/

#step in
cd ${SWORK}/run/

#datadir
datadir="/gpfs/alpinetds/scratch/tkurth/csc190/segm_h5_v3_uncompressed"
scratchdir="/xfs/scratch/"$(whoami)"/data"

#run
cat $LSB_DJOB_HOSTFILE | sort | uniq | grep -v login | grep -v batch > host_list
mpirun -np 1 --bind-to none -x PATH -x LD_LIBRARY_PATH --hostfile host_list -npernode 1 ./stage_in.sh ${datadir} ${scratchdir}
mpirun -np 1 --bind-to none -x PATH -x LD_LIBRARY_PATH --hostfile host_list -npernode 1 python ./mascarpone-tiramisu-tf.py --lr 1e-5 --datadir ${scratchdir}
#mpirun -np 24 --bind-to none -x PATH -x LD_LIBRARY_PATH --hostfile host_list -npernode 6 python ./test_hvd.py
