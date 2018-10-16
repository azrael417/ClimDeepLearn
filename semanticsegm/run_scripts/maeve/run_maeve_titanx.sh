#!/bin/bash

#load python env
source activate thorsten-tf-py27

#openmp stuff
export OMP_NUM_THREADS=6
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#pick GPU
export CUDA_VISIBLE_DEVICES=2

#directories
#datadir=/data1/tkurth/tiramisu/segm_h5_v3_new_split
datadir=/data1/tkurth/tiramisu/segm_h5_v3_old
#scratchdir=${DW_PERSISTENT_STRIPED_DeepCAM}/$(whoami)
scratchdir=${datadir}
numfiles_train=1500
numfiles_validation=300
numfiles_test=500

#create run dir
run_dir=/data1/tkurth/tiramisu/runs/test_run_nn256_np1536_j143026_convergence
#rundir=${WORK}/data/tiramisu/runs/run_nnodes16_j6415751
mkdir -p ${run_dir}

#copy relevant files
cp stage_in_parallel.sh ${run_dir}/
cp ../../utils/parallel_stagein.py ${run_dir}/
cp ../../utils/graph_flops.py ${run_dir}/
cp ../../utils/climseg_helpers.py ${run_dir}/
cp ../../deeplab-tf/deeplab-tf-lite-train.py ${run_dir}/
cp ../../deeplab-tf/deeplab-tf-lite-inference.py ${run_dir}/
cp ../../deeplab-tf/model.py ${run_dir}/
cp ../../deeplab-tf/model_helpers.py ${run_dir}/

#step in
cd ${run_dir}

#run the training

#fp32 lag 1, full
#srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 264 -u python ./deeplab-tf.py --datadir_train ${scratchdir}/train/data --datadir_validation ${scratchdir}/validation/data --chkpt_dir checkpoint.fp32.lag1 --epochs 4 --fs local --loss weighted_mean --cluster_loss_weight 0.0 --optimizer opt_type=LARC-Adam,learning_rate=0.0001,gradient_lag=1 --model=resnet_v2_50 --scale_factor 1.0 --batch 1 --decoder=deconv1x --device "/device:cpu:0" --data_format "channels_last" |& tee out.fp32.lag1.${SLURM_JOBID}

#some parameters
lag=0
train=0
test=1

if [ ${train} -eq 1 ]; then
  echo "Starting Training"
  python -u ./deeplab-tf-lite-train.py --datadir_train ${scratchdir}/train \
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
                                       --device "/device:cpu:0" \
                                       --dtype float16 \
				       --label_id 0 \
                                       --data_format "channels_last" |& tee out.fp16.lag${lag}.train
fi

if [ ${test} -eq 1 ]; then
  echo "Starting Testing"
  python -u ./deeplab-tf-lite-inference.py --datadir_test ${scratchdir}/test \
                                           --chkpt_dir checkpoint.fp16.lag${lag} \
					   --test_size 100 \
					   --output_graph deepcam_inference.pb \
                                           --output output_test_debug \
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
