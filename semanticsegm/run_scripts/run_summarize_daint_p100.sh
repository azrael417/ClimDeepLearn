#!/bin/bash
#SBATCH -J climseg_horovod
#SBATCH -A g107
#SBATCH -t 02:00:00
#SBATCH -p normal
#SBATCH -C gpu

#setting up stuff
module unload PrgEnv-cray
module load PrgEnv-gnu
module load gcc/5.3.0
module load cudatoolkit/8.0.61_2.4.3-6.0.4.0_3.1__gb475d12
module load cray-hdf5/1.10.0.3
source activate tensorflow-mpi4py

#openmp stuff
export OMP_NUM_THREADS=12
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export CRAY_CUDA_MPS=1
export MPICH_RDMA_ENABLED_CUDA=1

#directories
datadir=/scratch/snx3000/tkurth/data/tiramisu/segm_h5_v3_reformat

#run the training
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 24 python -u ../tiramisu-tf/summarize_stats.py --prefix="data" --input_path=${datadir} --output_path=${datadir}

