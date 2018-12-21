#!/bin/bash

#openmp stuff
export OMP_NUM_THREADS=6
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#pick GPU
export CUDA_VISIBLE_DEVICES=2

#directories
datadir=/mnt/data
scratchdir=/mnt/data
numfiles_train=1500
numfiles_validation=300
numfiles_test=500
downsampling=4
batch=8

#create run dir
run_dir=/mnt/runs/deeplab/run_ngpus1
#rundir=${WORK}/data/tiramisu/runs/run_nnodes16_j6415751
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
lag=0
train=0
test=1

if [ ${train} -eq 1 ]; then
  echo "Starting Training"
  runid=0
  runfiles=$(ls -latr out.lite.fp32.lag${lag}.train.run* | tail -n1 | awk '{print $9}')
  if [ ! -z ${runfiles} ]; then
      runid=$(echo ${runfiles} | awk '{split($1,a,"run"); print a[1]+1}')
  fi
    
  python -u ./deeplab-tf-train.py      --datadir_train ${scratchdir}/train \
                                       --train_size ${numfiles_train} \
                                       --datadir_validation ${scratchdir}/validation \
                                       --validation_size ${numfiles_validation} \
                                       --downsampling ${downsampling} \
				       --downsampling_mode "center-crop" \
                                       --channels 0 1 2 10 \
                                       --chkpt_dir checkpoint.fp32.lag${lag} \
                                       --epochs 50 \
                                       --fs local \
                                       --loss weighted_mean \
                                       --optimizer opt_type=LARC-Adam,learning_rate=0.0001,gradient_lag=${lag} \
                                       --model resnet_v2_50 \
                                       --scale_factor 1.0 \
                                       --batch ${batch} \
                                       --decoder bilinear \
                                       --device "/device:cpu:0" \
                                       --label_id 0 \
                                       --disable_imsave \
				       --use_batchnorm \
                                       --data_format "channels_last" |& tee out.lite.fp32.lag${lag}.train.run${runid}
fi

if [ ${test} -eq 1 ]; then
  echo "Starting Testing"
  runid=0
  runfiles=$(ls -latr out.lite.fp32.lag${lag}.test.run* | tail -n1 | awk '{print $9}')
  if [ ! -z ${runfiles} ]; then
      runid=$(echo ${runfiles} | awk '{split($1,a,"run"); print a[1]+1}')
  fi
    
  python -u ./deeplab-tf-inference.py      --datadir_test ${scratchdir}/test \
                                           --test_size ${numfiles_test} \
                                           --downsampling ${downsampling} \
					   --downsampling_mode "center-crop" \
                                           --channels 0 1 2 10 \
                                           --chkpt_dir checkpoint.fp32.lag${lag} \
                                           --output_graph deepcam_inference.pb \
                                           --output output_test \
                                           --fs local \
                                           --loss weighted_mean \
                                           --model=resnet_v2_50 \
                                           --scale_factor 1.0 \
                                           --batch ${batch} \
                                           --decoder bilinear \
                                           --device "/device:cpu:0" \
                                           --label_id 0 \
					   --use_batchnorm \
                                           --data_format "channels_last" |& tee out.lite.fp32.lag${lag}.test.run${runid}
fi
