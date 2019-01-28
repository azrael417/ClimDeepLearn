#!/bin/bash

#openmp stuff
export OMP_NUM_THREADS=6
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#pick GPU
export CUDA_VISIBLE_DEVICES=0

#directories and files
datadir=/mnt/data
scratchdir=/mnt/data
numfiles_train=1500
numfiles_validation=300
numfiles_test=500

#network parameters
downsampling=4
batch=8
#blocks="2 2 2 4 5"
blocks="3 3 4 4 7 7"

#create run dir
run_dir=/mnt/runs/tiramisu/run2_ngpus1
#rundir=${WORK}/data/tiramisu/runs/run_nnodes16_j6415751
mkdir -p ${run_dir}

#copy relevant files
cp stage_in_parallel.sh ${run_dir}/
cp ../../utils/parallel_stagein.py ${run_dir}/
cp ../../utils/graph_flops.py ${run_dir}/
cp ../../utils/common_helpers.py ${run_dir}/
cp ../../utils/data_helpers.py ${run_dir}/
cp ../../tiramisu-tf/tiramisu-tf-train.py ${run_dir}/
cp ../../tiramisu-tf/tiramisu-tf-inference.py ${run_dir}/
cp ../../tiramisu-tf/tiramisu_model.py ${run_dir}/

#step in
cd ${run_dir}

#some parameters
lag=0
train=1
test=1

if [ ${train} -eq 1 ]; then
  echo "Starting Training"
  runid=0
  runfiles=$(ls -latr out.lite.fp32.lag${lag}.train.run* | tail -n1 | awk '{print $9}')
  if [ ! -z ${runfiles} ]; then
      runid=$(echo ${runfiles} | awk '{split($1,a,"run"); print a[1]+1}')
  fi
    
  python -u ./tiramisu-tf-train.py      --datadir_train ${scratchdir}/train \
                                        --train_size ${numfiles_train} \
                                        --datadir_validation ${scratchdir}/validation \
                                        --validation_size ${numfiles_validation} \
                                        --chkpt_dir checkpoint.fp32.lag${lag} \
					--downsampling ${downsampling} \
                                        --downsampling_mode "center-crop" \
                                        --disable_imsave \
                                        --epochs 20 \
                                        --fs local \
                                        --channels 0 1 2 10 \
                                        --blocks ${blocks} \
                                        --growth 32 \
                                        --filter-sz 5 \
                                        --loss weighted \
                                        --optimizer opt_type=LARC-Adam,learning_rate=0.0001,gradient_lag=${lag} \
                                        --scale_factor 1.0 \
                                        --batch ${batch} \
					--use_batchnorm \
                                        --label_id 0 \
					--data_format "channels_first" |& tee out.lite.fp32.lag${lag}.train.run${runid}
fi

if [ ${test} -eq 1 ]; then
  echo "Starting Testing"
  runid=0
  runfiles=$(ls -latr out.lite.fp32.lag${lag}.test.run* | tail -n1 | awk '{print $9}')
  if [ ! -z ${runfiles} ]; then
      runid=$(echo ${runfiles} | awk '{split($1,a,"run"); print a[1]+1}')
  fi
    
  python -u ./tiramisu-tf-inference.py     --datadir_test ${scratchdir}/test \
                                           --test_size ${numfiles_test} \
                                           --downsampling ${downsampling} \
                                           --downsampling_mode "center-crop" \
                                           --channels 0 1 2 10 \
                                           --chkpt_dir checkpoint.fp32.lag${lag} \
					   --output_graph tiramisu_inference.pb \
                                           --output output_test \
                                           --fs local \
					   --blocks ${blocks} \
					   --growth 32 \
					   --filter-sz 5 \
                                           --loss weighted \
                                           --scale_factor 1.0 \
                                           --batch ${batch} \
					   --use_batchnorm \
                                           --label_id 0 \
                                           --data_format "channels_first" |& tee out.lite.fp32.lag${lag}.test.run${runid}
fi
