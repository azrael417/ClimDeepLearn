#!/bin/bash
#SBATCH -q regular
#SBATCH -A m1759
#SBATCH -C knl
#SBATCH -t 6:00:00
#SBATCH -J mascarpone_climsegment_horovod

#set up python stuff
module load python
source activate thorstendl-horovod
module load gcc/6.3.0

#add this to library path:
#modulebase=$(dirname $(module show tensorflow/intel-head 2>&1 | grep PATH |awk '{print $3}'))
#export PYTHONPATH=${modulebase}/lib/python2.7/site-packages:${PYTHONPATH}

#set up run directory
run_dir=${WORK}/gb2018/tiramisu/runs/cori/run_j${SLURM_JOBID}
mkdir -p ${run_dir}
cp ../deeplab-tf/deeplab-tf.py ${run_dir}/
cp ../deeplab-tf/tiramisu_helpers.py ${run_dir}/

#step in
cd ${run_dir}

#other directories
datadir=${WORK}/gb2018/tiramisu/segm_h5_v3_reformat

#run the training
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 272 -u python -u deeplab-tf.py --fs local --datadir ${scratchdir} --cluster_loss_weight 0.0 --lr 1e-4 --optimizer="LARC-Adam"