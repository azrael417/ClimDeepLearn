#!/bin/bash
#SBATCH -J climseg_horovod
#SBATCH -t 04:00:00
#SBATCH -A g107
#SBATCH -p normal
#SBATCH -C gpu 

#setting up stuff
module unload PrgEnv-cray
module load PrgEnv-gnu
module load gcc/5.3.0
module load cudatoolkit/8.0.61_2.4.3-6.0.4.0_3.1__gb475d12
module load cray-hdf5/1.10.0.3
source activate tensorflow

#openmp stuff
export OMP_NUM_THREADS=12
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export CRAY_CUDA_MPS=0
export MPICH_RDMA_ENABLED_CUDA=1

#directories
datadir=/scratch/snx3000/tkurth/data/tiramisu/segm_h5_v3_reformat
#scratchdir=/tmp/tiramisu

#create run dir
rundir=${WORK}/data/tiramisu/runs/run_nnodes${SLURM_NNODES}_j${SLURM_JOBID}
mkdir -p ${rundir}
cp ../tiramisu-tf/tiramisu_helpers.py ${rundir}/
cp ../tiramisu-tf/mascarpone-tiramisu-tf-singlefile.py ${rundir}/

#step in
cd ${rundir}

#run the training
#srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 24 python -u ./stage_in_2.sh ${datadir} ${scratchdir} 
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 24 -u python -u ./mascarpone-tiramisu-tf-singlefile.py --fs global --datadir ${datadir} --blocks 3 3 4 7 10 --loss weighted --lr 1e-5 |& tee out.${SLURM_JOBID}
