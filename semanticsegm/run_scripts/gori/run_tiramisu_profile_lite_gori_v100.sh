#!/bin/bash
#SBATCH -J climseg_horovod
#SBATCH -t 02:00:00
#SBATCH -A nstaff
#SBATCH -C gpu
#SBATCH --gres=gpu:8
#SBATCH --gres-flags=enforce-binding
#SBATCH --exclusive

#load modules
module load cuda
module load nccl
module load python3/3.6-anaconda-4.4

#activate env
source activate thorstendl-gori-py3-tf

#rankspernode
rankspernode=1

#openmp stuff
export OMP_NUM_THREADS=$(( 40 / ${rankspernode} ))
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
sruncmd="srun --mpi=pmi2 -N ${SLURM_NNODES} -n $(( ${SLURM_NNODES} * ${rankspernode} )) -c $(( 80 / ${rankspernode} )) --cpu_bind=cores"

#directories and files
datadir=/project/projectdirs/mpccc/tkurth/DataScience/gb2018/data/segm_h5_v3_new_split_maeve
scratchdir=${datadir}
numfiles_train=1500
numfiles_validation=300
numfiles_test=500

#network parameters
downsampling=4
batch=8
blocks="2 2 2 4 5"
#blocks="3 3 4 4 7 7"

#create run dir
run_dir=/project/projectdirs/mpccc/tkurth/DataScience/gb2018/runs/run2_ngpus1
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
test=0
prec=32

#list of metrics
#metrics="time"
metrics="smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
lts__t_sectors_aperture_sysmem_op_read.sum,lts__t_sectors_aperture_sysmem_op_write.sum,\
dram__sectors_read.sum,dram__sectors_write.sum,\
lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum"

if [ ${train} -eq 1 ]; then
    for metric in ${metrics}; do
      metricname=${metric//,/-}
      echo "Starting Training Profiling for Metric ${metric}"
      if [ "${metric}" == "time" ]; then
	  profilestring="nv-nsight-cu-cli"
      else
	  profilestring="nv-nsight-cu-cli --metrics ${metric}"
      fi
      runid=0
      runfiles=$(ls -latr out.lite.fp${prec}.lag${lag}.train.allmetrics.run*.profile | tail -n1 | awk '{print $9}')
      if [ ! -z ${runfiles} ]; then
	  runid=$(echo ${runfiles} | awk '{split($1,a,"run"); print a[1]+1}')
      fi

      #set profile string
      profilestring=${profilestring}" -f -o out.lite.fp${prec}.lag${lag}.train.allmetrics.run${runid}.profile"
      #profilestring=
  
      ${sruncmd} ${profilestring} $(which python) -u ./tiramisu-tf-train.py      --datadir_train ${scratchdir}/train \
                                                                        --train_size $(( ${batch} * 3 )) \
	                                                                --datadir_validation ${scratchdir}/validation \
							                --validation_size ${numfiles_validation} \
	                                                                --disable_checkpoints \
							                --chkpt_dir checkpoint.fp${prec}.lag${lag} \
							                --downsampling ${downsampling} \
							                --downsampling_mode "center-crop" \
							                --disable_imsave \
							                --epochs 1 \
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
	                                                                --dtype float${prec} \
							                --data_format "channels_first" |& tee out.lite.fp${prec}.lag${lag}.train.allmetrics.run${runid}
      exit
    done
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
