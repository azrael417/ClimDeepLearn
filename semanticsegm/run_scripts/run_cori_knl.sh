#!/bin/bash
#SBATCH -q regular
#SBATCH -A m1759
#SBATCH -C knl
#SBATCH -t 6:00:00
#SBATCH -J climsegment_horovod

#set up python stuff
module load python
source activate thorstendl-horovod
module load gcc/6.3.0

#add this to library path:
modulebase=$(dirname $(module show tensorflow/intel-head 2>&1 | grep PATH |awk '{print $3}'))
export PYTHONPATH=${modulebase}/lib/python2.7/site-packages:${PYTHONPATH}

#create and copy everything to the run directory
mkdir -p ${WORK}/gb2018/tiramisu/runs/j${SLURM_JOB_ID}
cp ../tiramisu-tf/mascarpone-tiramisu-tf-singlefile.py ${WORK}/gb2018/tiramisu/runs/j${SLURM_JOB_ID}

#step in
cd ${WORK}/gb2018/tiramisu/runs/j${SLURM_JOB_ID}

#run the training
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 272 -u python -u ./mascarpone-tiramisu-tf-singlefile.py --datadir /global/cscratch1/sd/tkurth/gb2018/tiramisu/segm_h5_v3_reformat
#srun -N 1 -n 1 -c 272 -u python -u ../tiramisu-tf/tiramisu-tf.py
