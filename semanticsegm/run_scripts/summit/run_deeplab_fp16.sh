#!/bin/bash
# Disable multiple threads
export OMPI_MCA_osc_pami_allow_thread_multiple=0

#disable adaptive routing
export PAMI_IBV_ENABLE_OOO_AR=0
export PAMI_IBV_QP_SERVICE_LEVEL=0

# Reduce horovod sleep time, enable priority NCCL stream
export HOROVOD_SLEEP_INTERVAL=2
export HOROVOD_USE_PRIORITY=0

VENV=pyvenv_summit_7.5.18
source ${1}/${VENV}/bin/activate

grank=$PMIX_RANK
lrank=$(($PMIX_RANK%6))

<<<<<<< HEAD:semanticsegm/run_scripts/bigruns_update/run_deeplab.sh
APP="python ./deeplab-tf.py  --datadir_train ${1}/train/data --datadir_validation ${1}/validation/data --epochs ${2} --fs local --loss weighted --cluster_loss_weight 0.0  --optimizer opt_type=LARC-Adam,learning_rate=${3},gradient_lag=${5} --model=resnet_v2_50 --scale_factor ${4} --batch ${6} --use_batchnorm"
=======
APP="python ./deeplab-tf.py  --datadir ${1}/data/ --epochs ${2} --fs local --loss weighted --cluster_loss_weight 0.0 --optimizer opt_type=LARC-Adam,learning_rate=${3},gradient_lag=${5} --model=resnet_v2_50 --batch ${6} --dtype float16 --scale_factor ${4}"
>>>>>>> c18f2c2ac787da057fee0dabcaae00efb7d5a508:semanticsegm/run_scripts/summit/run_deeplab_fp16.sh

export PAMI_ENABLE_STRIPING=0

case ${lrank} in
[0])
export PAMI_IBV_DEVICE_NAME=mlx5_0:1
#numactl --physcpubind=0,4,8,12,16,20,24 --membind=0 $APP
numactl --physcpubind=0-27 --membind=0 $APP
  ;;
[1])
export PAMI_IBV_DEVICE_NAME=mlx5_1:1
#numactl --physcpubind=28,32,36,40,44,48,52 --membind=0 $APP
numactl --physcpubind=28-55 --membind=0 $APP
  ;;
[2])
export PAMI_IBV_DEVICE_NAME=mlx5_0:1
#numactl --physcpubind=56,60,64,68,72,76,80 --membind=0 $APP
numactl --physcpubind=56-83 --membind=0 $APP
  ;;
[3])
export PAMI_IBV_DEVICE_NAME=mlx5_3:1
#numactl --physcpubind=88,92,96,100,104,108,112 --membind=8 $APP
numactl --physcpubind=88-115 --membind=8 $APP
  ;;
[4])
export PAMI_IBV_DEVICE_NAME=mlx5_2:1
#numactl --physcpubind=116,120,124,128,132,136,140 --membind=8 $APP
numactl --physcpubind=116-143 --membind=8 $APP
  ;;
[5])
export PAMI_IBV_DEVICE_NAME=mlx5_3:1
#numactl --physcpubind=144,148,152,156,160,164,168 --membind=8 $APP
numactl --physcpubind=144-171 --membind=8 $APP
  ;;
esac