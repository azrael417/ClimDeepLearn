#!/bin/bash
#SBATCH -J climseg_horovod
#SBATCH -t 02:00:00
#SBATCH -A nstaff
#SBATCH -C gpu
#SBATCH --gres=gpu:8
#SBATCH --gres-flags=enforce-binding
#SBATCH --exclusive

#set up python stuff
module load cuda
module load nccl
module load gcc/7.3.0
module load openmpi/3.1.0-ucx
module load python3/3.6-anaconda-4.4
source activate thorstendl-gori-py3-tf 

#rankspernode
rankspernode=8

#openmp stuff
export OMP_NUM_THREADS=$(( 40 / ${rankspernode} ))
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export HDF5_USE_FILE_LOCKING=FALSE
sruncmd="srun -u --mpi=pmi2 -N ${SLURM_NNODES} -n $(( ${SLURM_NNODES} * ${rankspernode} )) -c $(( 80 / ${rankspernode} )) --cpu_bind=cores"

#directories
datadir=/project/projectdirs/mpccc/tkurth/DataScience/gb2018/data/segm_h5_v3_new_split_maeve
#datadir=/data1/mudigonda/missing_files_for_gb_video
#scratchdir=${DW_PERSISTENT_STRIPED_DeepCAM}/$(whoami)
#scratchdir=/dev/shm/tkurth/deepcam/data
scratchdir=${datadir}
numfiles_train=1500
numfiles_validation=300
numfiles_test=500

#create run dir
#run_dir=/data1/tkurth/deeplab/runs/run_1
run_dir=/project/projectdirs/mpccc/tkurth/DataScience/gb2018/runs/deeplab/run1_ngpus1

rm -rf ${run_dir}/*
mkdir -p ${run_dir}

#copy relevant files
cp stage_in_parallel.sh ${run_dir}/
cp ../../utils/parallel_stagein.py ${run_dir}/
cp ../../utils/graph_flops.py ${run_dir}/
cp ../../utils/common_helpers.py ${run_dir}/
cp ../../utils/data_helpers.py ${run_dir}/
cp ../../deeplab-tf/deeplab-tf-train.py ${run_dir}/
cp ../../deeplab-tf/deeplab-tf-inference.py ${run_dir}/
cp ../../deeplab-tf/deeplab_model.py ${run_dir}/

#step in
cd ${run_dir}

#some parameters
#operation mode
stage=1
train=1
test=0
#network
lag=1
prec=16
batch=2
scale_factor=0.1

#stage in
if [ "${scratchdir}" != "${datadir}" ]; then
    if [ ${stage} -eq 1 ]; then
	cmd="srun --mpi=pmi2 -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 80 ./stage_in_parallel.sh ${datadir} ${scratchdir} ${numfiles_train} ${numfiles_validation} ${numfiles_test}"
	echo ${cmd}
	${cmd}
    fi
else
    echo "Scratchdir and datadir is the same, no staging needed!"
fi

#train
if [ ${train} -eq 1 ]; then
  echo "Starting Training"
  ${sruncmd} python -u ./deeplab-tf-train.py      --datadir_train ${scratchdir}/train \
                                       --train_size ${numfiles_train} \
                                       --datadir_validation ${scratchdir}/validation \
                                       --validation_size ${numfiles_validation} \
                                       --chkpt_dir checkpoint.fp${prec}.lag${lag} \
                                       --disable_checkpoint \
                                       --epochs 20 \
                                       --fs global \
                                       --loss weighted \
                                       --optimizer opt_type=LARC-Adam,learning_rate=0.0001,gradient_lag=${lag} \
                                       --model "resnet_v2_50" \
                                       --scale_factor ${scale_factor} \
                                       --batch ${batch} \
                                       --decoder "deconv1x" \
                                       --device "/device:cpu:0" \
                                       --dtype "float${prec}" \
				       --label_id 0 \
                                       --data_format "channels_first" |& tee out.fp${prec}.lag${lag}.train
fi

if [ ${test} -eq 1 ]; then
  echo "Starting Testing"
  ${sruncmd} python -u ./deeplab-tf-inference.py      --datadir_test ${scratchdir}/test \
                                           --chkpt_dir checkpoint.fp${prec}.lag${lag} \
					   --test_size -1 \
					   --output_graph deepcam_inference.pb \
                                           --output output_test_5 \
                                           --fs local \
                                           --loss weighted \
                                           --model "resnet_v2_50" \
                                           --scale_factor ${scale_factor} \
                                           --batch ${batch} \
                                           --decoder "deconv1x" \
                                           --device "/device:cpu:0" \
                                           --dtype "float${prec}" \
					   --label_id 0 \
                                           --data_format "channels_last" |& tee out.fp${prec}.lag${lag}.test
fi
