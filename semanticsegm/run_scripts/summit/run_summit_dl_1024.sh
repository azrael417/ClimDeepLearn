#!/bin/bash
##BSUB -jsm d
#BSUB -W 240
#BSUB -P CSC275PRABHAT
#BSUB -alloc_flags "nvme"
#BSUB -J 1024_dl_lr0.01
#BSUB -o out_test.%J
#BSUB -e out_test.%J
#BSUB -q tested
#BSUB -csm y
#BSUB -R "1*{select[LN]span[hosts=1]} + 43008*{select[CN && (hname != 'e28n15') && (hname != 'e01n16') && (hname != 'e01n12') && (hname != 'f05n14') && (hname != 'f17n08') && (hname != 'f17n09') && (hname != 'f17n10') && (hname != 'f17n11') && (hname != 'f17n12') && (hname != 'f17n13') && (hname != 'f17n14') && (hname != 'f17n15') && (hname != 'f17n16') && (hname != 'f17n17') && (hname != 'f17n18') && (hname != 'f18n08') && (hname != 'f18n09') && (hname != 'f18n10') && (hname != 'f18n11') && (hname != 'f18n12') && (hname != 'f18n13') && (hname != 'f18n14') && (hname != 'f18n15') && (hname != 'f18n16') && (hname != 'f18n17') && (hname != 'f18n18') && (hname != 'f19n08') && (hname != 'f19n09') && (hname != 'f19n10') && (hname != 'f19n11') && (hname != 'f19n12') && (hname != 'f19n13') && (hname != 'f19n14') && (hname != 'f19n15') && (hname != 'f19n16') && (hname != 'f19n17') && (hname != 'f19n18') && (hname != 'f20n08') && (hname != 'f20n09') && (hname != 'f20n10') && (hname != 'f20n11') && (hname != 'f20n12') && (hname != 'f20n13') && (hname != 'f20n14') && (hname != 'f20n15') && (hname != 'f20n16') && (hname != 'f20n17') && (hname != 'f20n18') && (hname != 'f21n08') && (hname != 'f21n09') && (hname != 'f21n10') && (hname != 'f21n11') && (hname != 'f21n12') && (hname != 'f21n13') && (hname != 'f21n14') && (hname != 'f21n15') && (hname != 'f21n16') && (hname != 'f21n17') && (hname != 'f21n18') && (hname != 'f22n08') && (hname != 'f22n09') && (hname != 'f22n10') && (hname != 'f22n11') && (hname != 'f22n12') && (hname != 'f22n13') && (hname != 'f22n14') && (hname != 'f22n15') && (hname != 'f22n16') && (hname != 'f22n17') && (hname != 'f22n18') && (hname != 'f23n08')]order[!-slots:maxslots]span[ptile=42] }"

#load modules
source run_env.sh

#jsrun debug flags
jsrundebug="--stdio_mode prepended_with_host --progress ~/progressfiles/progress."${LSB_JOBID}

#determine number of nodes and total procs
nprocspn=6
nnodes=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)
nprocs=$(( ${nnodes} * ${nprocspn} ))

#script in place
SWORK=/gpfs/alpinetds/world-shared/ven201/seant/climate/test_runs/convergence
#run_dir=${SWORK}/experimental_test_run_nn${nnodes}_np${nprocs}_j${LSB_JOBID}
run_dir=${SWORK}/experimental_test_run_nn1024_np6144_j146043
mkdir -p ${run_dir}

cp /ccs/home/tkurth/nodetest/run_test.sh ${run_dir}/
cp /ccs/home/tkurth/nodetest/test ${run_dir}/
cp stage_in_parallel.sh ${run_dir}/
cp run_deeplab_deconv.sh ${run_dir}/
cp run_deeplab_deconv_fp16.sh ${run_dir}/
cp ../../utils/parallel_stagein.py ${run_dir}/
cp ../../utils/graph_flops.py ${run_dir}/
cp ../../utils/climseg_helpers.py ${run_dir}/
cp ../../deeplab-tf/deeplab-tf.py ${run_dir}/

#step in
cd ${run_dir}

#datadir
datadir="/gpfs/alpinetds/world-shared/csc275/climdata_new_split"
scratchdir="/mnt/bb/"$(whoami)""
nperproc_train=250
nperproc_validation=25
numfiles_train=$(( ${nprocspn} * ${nperproc_train} ))
numfiles_validation=$(( ${nprocspn} * ${nperproc_validation} ))

#run
cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch > host_list

#nodetest
#echo "starting run_test.sh" `date`
#jsrun -n ${nnodes} -g 6 -c 42 -a ${nprocspn} --bind=proportional-packed:7 --launch_distribution=packed stdbuf -o0 ./run_test.sh
#echo "finished run_test.sh" `date`

#stagein
echo "starting stage_in_parallel.sh" `date`
jsrun ${jsrundebug} -n ${nnodes} -g 1 -c 42 -a 1 ./stage_in_parallel.sh ${datadir} ${scratchdir} ${numfiles_train} ${numfiles_validation}
echo "finished stage_in_parallel.sh" `date`

# Set flag after stage-in to prepend to spectrum-mpi's existing LD_PRELOAD
export OMPI_LD_PRELOAD_PREPEND="${scratchdir}/${VENV}/lib/directconv.so"

#do perftest
#echo "starting run_deeplab_deconv_perftest.sh" `date`
#jsrun -n ${nnodes} -g 6 -c 42 -a ${nprocspn} --bind=proportional-packed:7 --launch_distribution=packed stdbuf -o0 ./run_deeplab_deconv_perftest.sh ${scratchdir}
#echo "finished run_mascarpone_perftest.sh" `date`

#clean up crap
rm shuffle_indices.npy

##fp16-lag0
#echo "starting run_deeplab_deconv_fp16.sh" `date`
#jsrun ${jsrundebug} -n ${nnodes} -g 6 -c 42 -a ${nprocspn} --bind=proportional-packed:7 --launch_distribution=packed stdbuf -o0 ./run_deeplab_deconv_fp16.sh ${scratchdir} 400 0.01 0.1 0 |& tee out.fp16.lag0.${LSB_JOBID}
#echo "finished run_deeplab_deconv_fp16.sh" `date`

##fp32-lag0
#echo "starting run_deeplab_deconv.sh" `date`
#jsrun ${jsrundebug} -n ${nnodes} -g 6 -c 42 -a ${nprocspn} --bind=proportional-packed:7 --launch_distribution=packed stdbuf -o0 ./run_deeplab_deconv.sh ${scratchdir} 200 0.0064 0.1 0 |& tee #out.fp32.lag0.${LSB_JOBID}
#echo "finished run_deeplab_deconv.sh" `date`

#fp16-lag1
echo "starting run_deeplab_deconv_fp16.sh" `date`
jsrun ${jsrundebug} -n ${nnodes} -g 6 -c 42 -a ${nprocspn} --bind=proportional-packed:7 --launch_distribution=packed stdbuf -o0 ./run_deeplab_deconv_fp16.sh ${scratchdir} 200 0.0064 0.1 1 |& tee out.fp16.lag1.${LSB_JOBID}
echo "finished run_deeplab_deconv_fp16.sh" `date`

##fp32-lag1
#echo "starting run_deeplab_deconv.sh" `date`
#jsrun ${jsrundebug} -n ${nnodes} -g 6 -c 42 -a ${nprocspn} --bind=proportional-packed:7 --launch_distribution=packed stdbuf -o0 ./run_deeplab_deconv.sh ${scratchdir} 200 0.0064 0.1 1 |& tee out.fp32.lag1.${LSB_JOBID}
#echo "finished run_deeplab_deconv.sh" `date`
