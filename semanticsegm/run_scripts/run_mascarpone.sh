#!/bin/bash
# Disable multiple threads
export OMPI_MCA_osc_pami_allow_thread_multiple=0

VENV=pyvenv_TF-1.5_ompi_exp
source ~/${VENV}/bin/activate

# Modify LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/sw/summit/cuda/9.1.85/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=~/Summit/nccl_2.2.10/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=~/hdf5-1.10.1/lib:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=~/${VENV}/lib/python2.7/site-packages/horovod/tensorflow:$LD_LIBRARY_PATH

grank=$PMIX_RANK
lrank=$(($PMIX_RANK%6))

APP="python ./mascarpone-tiramisu-tf-singlefile.py --datadir /xfs/scratch/mfatica/data --blocks 3 3 4 7 10 --loss weighted --lr 1e-5"

#$APP

export PAMI_ENABLE_STRIPING=0

case ${lrank} in
[0])
export PAMI_IBV_DEVICE_NAME=mlx5_0:1
export OMPI_MCA_btl_openib_if_include=mlx5_0:1
numactl --physcpubind=0,4,8,12,16,20,24 --membind=0 $APP
  ;;
[1])
export PAMI_IBV_DEVICE_NAME=mlx5_1:1
export OMPI_MCA_btl_openib_if_include=mlx5_1:1
numactl --physcpubind=28,32,36,40,44,48,52 --membind=0 $APP
  ;;
[2])
export PAMI_IBV_DEVICE_NAME=mlx5_1:1
export OMPI_MCA_btl_openib_if_include=mlx5_1:1
numactl --physcpubind=56,60,64,68,72,76,80 --membind=0 $APP
  ;;
[3])
export PAMI_IBV_DEVICE_NAME=mlx5_3:1
export OMPI_MCA_btl_openib_if_include=mlx5_3:1
numactl --physcpubind=88,92,96,100,104,108,112 --membind=8 $APP
  ;;
[4])
export PAMI_IBV_DEVICE_NAME=mlx5_3:1
export OMPI_MCA_btl_openib_if_include=mlx5_3:1
numactl --physcpubind=116,120,124,128,132,136,140 --membind=8 $APP
  ;;
[5])
export PAMI_IBV_DEVICE_NAME=mlx5_2:1
export OMPI_MCA_btl_openib_if_include=mlx5_2:1
numactl --physcpubind=144,148,152,156,160,164,168 --membind=8 $APP
  ;;
esac
