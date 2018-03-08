#!/bin/bash
#BSUB -W 30
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

#determine number of nodes and total procs
nprocspn=6
nnodes=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)
nprocs=$(( ${nnodes} * ${nprocspn} ))

#script in place
run_dir=${SWORK}/scaling/run_nn${nnodes}_np${nprocs}_j${LSB_JOBID}
mkdir -p ${run_dir}
cp stage_in.sh ${run_dir}/
cp ../tiramisu-tf/mascarpone-tiramisu-tf*.py ${run_dir}/

#step in
cd ${run_dir}

#datadir
datadir="/gpfs/alpinetds/scratch/tkurth/csc190/segm_h5_v3_reformat"
scratchdir="/xfs/scratch/"$(whoami)"/data"

#run
cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch > host_list
mpirun -np ${nnodes} --bind-to none -x PATH -x LD_LIBRARY_PATH --hostfile host_list -npernode 1 ./stage_in.sh ${datadir} ${scratchdir} 1000
mpirun -np ${nprocs} --bind-to none -x PATH -x LD_LIBRARY_PATH --hostfile host_list -npernode ${nprocspn} python ./mascarpone-tiramisu-tf-singlefile.py --lr 1e-4 --datadir ${scratchdir} |& tee out.${LSB_JOBID}
