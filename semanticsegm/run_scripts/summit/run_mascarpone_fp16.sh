#!/bin/bash
# Disable multiple threads
export OMPI_MCA_osc_pami_allow_thread_multiple=0

# Reduce horovod sleep time
export HOROVOD_SLEEP_INTERVAL=2

VENV=pyvenv_summit_v3
# LD_PRELOAD fast custom kernel
export LD_PRELOAD=${1}/${VENV}/lib/directconv.so
source ${1}/${VENV}/bin/activate

grank=$PMIX_RANK
lrank=$(($PMIX_RANK%6))

APP="python ./mascarpone-tiramisu-tf-singlefile.py  --datadir ${1}/data/ --epochs ${2} --fs local --blocks 2 2 2 4 5 --growth 32 --filter-sz 5 --loss weighted --cluster_loss_weight 0.0  --lr ${3} --optimizer=LARC-Adam --batch 2 --dtype float16 --scale_factor ${4} --gradient-lag ${5} --disable_imsave --disable_checkpoints"

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
