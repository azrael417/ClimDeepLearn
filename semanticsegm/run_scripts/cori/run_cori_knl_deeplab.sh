#!/bin/bash
#SBATCH -J climseg_horovod
#SBATCH -t 04:00:00
#SBATCH -A dasrepo
#SBATCH -q regular
#SBATCH -C knl
#SBATCH -S 2

##DW persistentdw name=DeepCAM


#setting up stuff
module unload PrgEnv-intel
module load PrgEnv-gnu
module load python/3.6-anaconda-4.4
#source activate thorstendl-cori-2.7-tf1.10
source activate thorstendl-cori-py3-tf
#module load tensorflow/intel-1.8.0-py27

#openmp stuff
export OMP_NUM_THREADS=66
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#debug
export MKLDNN_VERBOSE=0 #2 is very verbose

#directories
datadir=/global/cscratch1/sd/tkurth/gb2018/tiramisu/segm_h5_v3_new_split_maeve
#scratchdir=${DW_PERSISTENT_STRIPED_DeepCAM}/$(whoami)
scratchdir=/global/cscratch1/sd/tkurth/temp/deeplab
numfiles_train=100
numfiles_validation=10
numfiles_test=10

#create run dir
run_dir=${WORK}/gb2018/tiramisu/runs/cori/deeplab/run_nnodes${SLURM_NNODES}_j${SLURM_JOBID}
#rundir=${WORK}/data/tiramisu/runs/run_nnodes16_j6415751
mkdir -p ${run_dir}

#copy relevant files
cp stage_in_parallel.sh ${run_dir}/
cp ../../utils/parallel_stagein.py ${run_dir}/
cp ../../utils/graph_flops.py ${run_dir}/
cp ../../utils/tracehook.py ${run_dir}/
cp ../../utils/common_helpers.py ${run_dir}/
cp ../../utils/data_helpers.py ${run_dir}/
cp ../../deeplab-tf/deeplab-tf-train.py ${run_dir}/
cp ../../deeplab-tf/deeplab-tf-inference.py ${run_dir}/
cp ../../deeplab-tf/deeplab_model.py ${run_dir}/

#step in
cd ${run_dir}

#run the training
#stage in
cmd="srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 264 ./stage_in_parallel.sh ${datadir} ${scratchdir} ${numfiles_train} ${numfiles_validation} ${numfiles_test}"
echo ${cmd}
${cmd}


#some parameters
lag=0
train=1
test=0

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
                                       --downsampling 2 \
                                       --channels 0 1 2 10 \
                                       --chkpt_dir checkpoint.fp32.lag${lag} \
                                       --epochs 1 \
                                       --fs local \
                                       --loss weighted_mean \
                                       --optimizer opt_type=LARC-Adam,learning_rate=0.0001,gradient_lag=${lag} \
                                       --model=resnet_v2_50 \
                                       --scale_factor 1.0 \
                                       --batch 1 \
                                       --decoder=deconv1x \
                                       --device "/device:cpu:0" \
                                       --label_id 0 \
                                       --disable_imsave \
                                       --tracing="2:5" \
                                       --trace_dir="./" \
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
                                           --downsampling 2 \
                                           --channels 0 1 2 10 \
                                           --chkpt_dir checkpoint.fp32.lag${lag} \
                                           --output_graph deepcam_inference.pb \
                                           --output output_test \
                                           --fs local \
                                           --loss weighted_mean \
                                           --model=resnet_v2_50 \
                                           --scale_factor 1.0 \
                                           --batch 1 \
                                           --decoder=deconv1x \
                                           --device "/device:cpu:0" \
                                           --label_id 0 \
                                           --data_format "channels_last" |& tee out.lite.fp32.lag${lag}.test.run${runid}
fi
