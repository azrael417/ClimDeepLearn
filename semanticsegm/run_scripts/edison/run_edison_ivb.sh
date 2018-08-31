#!/bin/bash
#SBATCH -J climseg_horovod
#SBATCH -t 01:00:00
#SBATCH -A dasrepo
#SBATCH -q regular
#SBATCH -C ivybridge


#setting up stuff
module unload PrgEnv-intel
module load PrgEnv-gnu
module load gcc/7.1.0
module load cray-hdf5
module load python/3.6-anaconda-5.2
source activate thorstendl-edison-2.7

#openmp stuff
export OMP_NUM_THREADS=12
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#directories
datadir=/global/cscratch1/sd/tkurth/gb2018/tiramisu/segm_h5_v3_new_split
scratchdir=/dev/shm/$(whoami)
numfiles_train=100
numfiles_validation=10

#create run dir
run_dir=${WORK}/gb2018/tiramisu/runs/edison/run_nnodes${SLURM_NNODES}_j${SLURM_JOBID}
#rundir=${WORK}/data/tiramisu/runs/run_nnodes16_j6415751
mkdir -p ${run_dir}

#copy relevant files
cp stage_in_parallel.sh ${run_dir}/
cp ../../utils/parallel_stagein.py ${run_dir}/
cp ../../utils/graph_flops.py ${run_dir}/
cp ../../utils/climseg_helpers.py ${run_dir}/
cp ../../deeplab-tf/deeplab-tf.py ${run_dir}/

#step in
cd ${run_dir}

#run the training
#stage in
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 24 ./stage_in_parallel.sh ${datadir} ${scratchdir} ${numfiles_train} ${numfiles_validation}

#fp32 lag 1
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 24 -u python ./deeplab-tf.py --datadir_train ${scratchdir}/train/data --datadir_validation ${scratchdir}/validation/data --chkpt_dir checkpoint.fp32.lag1 --epochs 4 --fs local --loss weighted --cluster_loss_weight 0.0 --optimizer opt_type=LARC-Adam,learning_rate=0.001,gradient_lag=0 --model=resnet_v2_50 --scale_factor 0.1 --batch 1 --decoder=deconv1x --device "/device:cpu:0" --data_format "channels_last" |& tee out.fp32.lag1.${SLURM_JOBID}
