#!/bin/bash
#SBATCH -J climseg_horovod
#SBATCH -t 02:00:00
#SBATCH -A dasrepo
#SBATCH -q regular
#SBATCH -C knl
#SBATCH -S 2


#setting up stuff
module unload PrgEnv-intel
module load PrgEnv-gnu
module swap gcc gcc/7.1.0
#module load python/3.6-anaconda-4.4
#source activate thorstendl-cori-2.7
module load tensorflow/intel-1.8.0-py27

#openmp stuff
export OMP_NUM_THREADS=66
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#directories
datadir=/global/cscratch1/sd/tkurth/gb2018/tiramisu/segm_h5_v3_new_split
scratchdir=/dev/shm/$(whoami)
numfiles_train=200
numfiles_validation=10

#create run dir
run_dir=${WORK}/gb2018/tiramisu/runs/cori/run_nnodes${SLURM_NNODES}_j${SLURM_JOBID}
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
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 264 ./stage_in_parallel.sh ${datadir} ${scratchdir} ${numfiles_train} ${numfiles_validation}

#fp32 lag 1
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 264 -u python ./deeplab-tf.py --datadir_train ${scratchdir}/train/data --datadir_validation ${scratchdir}/validation/data --chkpt_dir checkpoint.fp32.lag1 --epochs 4 --fs local --loss weighted_mean --cluster_loss_weight 0.0 --optimizer opt_type=LARC-Adam,learning_rate=0.0001,gradient_lag=1 --model=resnet_v2_50 --scale_factor 1.0 --batch 1 --decoder=deconv1x --device "/device:cpu:0" --data_format "channels_last" |& tee out.fp32.lag1.${SLURM_JOBID}
