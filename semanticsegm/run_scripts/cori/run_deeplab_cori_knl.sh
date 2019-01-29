#!/bin/bash
#SBATCH -J climseg_horovod
#SBATCH -t 02:00:00
#SBATCH -A nstaff
#SBATCH -C knl
#SBATCH -q regular

#load modules
module load python3/3.6-anaconda-4.4

#activate env
source activate thorstendl-cori-py3-tf

#rankspernode
rankspernode=1

#openmp stuff
export OMP_NUM_THREADS=$(( 136 / ${rankspernode} ))
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export HDF5_USE_FILE_LOCKING=FALSE
sruncmd="srun -u -N ${SLURM_NNODES} -n $(( ${SLURM_NNODES} * ${rankspernode} )) -c $(( 272 / ${rankspernode} )) --cpu_bind=sockets"

#directories
#datadir=/project/projectdirs/mpccc/tkurth/DataScience/gb2018/data/segm_h5_v3_new_split_maeve
datadir=/global/cscratch1/sd/tkurth/gb2018/tiramisu/segm_h5_v3_new_split_maeve
#scratchdir=${DW_PERSISTENT_STRIPED_DeepCAM}/$(whoami)
scratchdir=/dev/shm/tkurth/deepcam/data
numfiles_train=500
numfiles_validation=200
numfiles_test=300

#create run dir
#run_dir=/data1/tkurth/deeplab/runs/run_1
run_dir=/project/projectdirs/mpccc/tkurth/DataScience/gb2018/runs/deeplab/run1_ngpus1

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

#stage in
cmd="srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 272 --cpu_bind=cores ./stage_in_parallel.sh ${datadir} ${scratchdir} ${numfiles_train} ${numfiles_validation} ${numfiles_test}"
echo ${cmd}
${cmd}
exit

#some parameters
lag=0
train=1
test=0

#run the stuff
if [ ${train} -eq 1 ]; then
  echo "Starting Training"
  ${sruncmd} python -u ./deeplab-tf-train.py      --datadir_train ${scratchdir}/train \
                                       --train_size ${numfiles_train} \
                                       --datadir_validation ${scratchdir}/validation \
                                       --validation_size ${numfiles_validation} \
                                       --chkpt_dir checkpoint.fp16.lag${lag} \
                                       --epochs 20 \
                                       --fs local \
                                       --loss weighted_mean \
                                       --optimizer opt_type=LARC-Adam,learning_rate=0.0001,gradient_lag=${lag} \
                                       --model=resnet_v2_50 \
                                       --scale_factor 1.0 \
                                       --batch 2 \
                                       --decoder deconv1x \
                                       --dtype float16 \
				       --label_id 0 \
                                       --data_format "channels_last" |& tee out.fp16.lag${lag}.train
fi

if [ ${test} -eq 1 ]; then
  echo "Starting Testing"
  ${sruncmd} python -u ./deeplab-tf-inference.py      --datadir_test ${scratchdir}/test \
                                           --chkpt_dir checkpoint.fp16.lag${lag} \
					   --test_size -1 \
					   --output_graph deepcam_inference.pb \
                                           --output output_test_5 \
                                           --fs local \
                                           --loss weighted_mean \
                                           --model=resnet_v2_50 \
                                           --scale_factor 1.0 \
                                           --batch 2 \
                                           --decoder deconv1x \
                                           --device "/device:cpu:0" \
                                           --dtype float16 \
					   --label_id 0 \
                                           --data_format "channels_last" |& tee out.fp16.lag${lag}.test
fi
