#!/bin/bash
#BSUB -nnodes 64
#BSUB -W 30
##BSUB -P CSC275PRABHAT
#BSUB -P VEN101
#BSUB -alloc_flags "smt4 nvme"
#BSUB -J climseg_training
#BSUB -o out_test.%J
#BSUB -e out_test.%J
#BSUB -q batch

# load modules
module load cuda

#determine number of nodes and total procs
nprocspn=6
nnodes=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)
nprocs=$(( ${nnodes} * ${nprocspn} ))

#script in place
SWORK=/gpfs/alpinetds/world-shared/ven201/seant/climate/test_runs
run_dir=${SWORK}/test_run_nn${nnodes}_np${nprocs}_j${LSB_JOBID}
mkdir -p ${run_dir}

cp stage_in_parallel.sh ${run_dir}/
cp run_mascarpone.sh ${run_dir}/
cp run_mascarpone_fp16.sh ${run_dir}/
cp ../../tiramisu-tf/parallel_stagein.py ${run_dir}/
cp ../../tiramisu-tf/mascarpone-tiramisu-tf*.py ${run_dir}/
cp ../../tiramisu-tf/tiramisu_helpers.py ${run_dir}/

#step in
cd ${run_dir}

#datadir
datadir="/gpfs/alpinetds/world-shared/csc275/climdata_reformat"
scratchdir="/mnt/bb/"$(whoami)""
nperproc=250
numfiles=$(( ${nprocspn} * ${nperproc} )) 

#run
cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch > host_list

echo "starting stage_in_parallel.sh " `date`
jsrun -n ${nnodes} -g 1 -c 42 -a 1 ./stage_in_parallel.sh ${datadir} ${scratchdir} ${numfiles}
echo "finished stage_in_parallel.sh" `date`

# Set flag after stage-in to prepend to spectrum-mpi's existing LD_PRELOAD
export OMPI_LD_PRELOAD_PREPEND="${scratchdir}/pyvenv_summit_7.5.18/lib/directconv.so"

echo "starting run_mascarpone.sh" `date`
# NOTE: arguments after scratchdir are (epochs, lr, scale_factor, gradient-lag)

#fp16-lag0
jsrun  -n ${nnodes} -g 6 -c 42 -a ${nprocspn} --bind=proportional-packed:7 --launch_distribution=packed stdbuf -o0 ./run_mascarpone_fp16.sh ${scratchdir} 10 0.0001 0.1 0 |& tee out.fp16.lag0.${LSB_JOBID}
echo "finished run_mascarpone.sh" `date`
