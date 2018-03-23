#!/bin/bash
#BSUB -nnodes 1024
#BSUB -W 60
#BSUB -P VEN101SUMMIT
#BSUB -alloc_flags "smt4 nvme"
#BSUB -J climseg_training
#BSUB -o out.%J
#BSUB -e err.%J
#BSUB -q tested

#determine number of nodes and total procs
nprocspn=6
nnodes=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)
nprocs=$(( ${nnodes} * ${nprocspn} ))


#script in place
SWORK=/gpfs/alpinetds/scratch/mfatica/ven101
run_dir=${SWORK}/SC_scaling/run_nn${nnodes}_np${nprocs}_j${LSB_JOBID}
#run_dir=${SWORK}/testing/run_nn${nnodes}_np${nprocs}_j${LSB_JOBID}
mkdir -p ${run_dir}

#cp stage_in_2_spectrum.sh ${run_dir}/
cp stage_in.sh ${run_dir}/
cp run_mascarpone.sh ${run_dir}/
cp ../tiramisu-tf/stagein.py ${run_dir}/
cp ../tiramisu-tf/mascarpone-tiramisu-tf*.py ${run_dir}/
cp ../tiramisu-tf/tiramisu_helpers.py ${run_dir}/

#step in
cd ${run_dir}

#datadir
#datadir="/gpfs/alpinetds/world-shared/csc275/climdata_reformat"
datadir="/gpfs/alpinetds/scratch/mfatica/ven101"
scratchdir="/xfs/scratch/"$(whoami)"/data"
nperproc=126
numfiles=$(( ${nprocspn} * ${nperproc} )) 

#run
cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch > host_list
#jsrun -n ${nnodes} -g 1 -c 42 -a 1 ./stage_in_2_spectrum.sh ${datadir} ${scratchdir} ${numfiles}

echo "starting stage_in.sh"
jsrun -n ${nnodes} -g 1 -c 42 -a 1 ./stage_in.sh ${datadir} ${scratchdir} ${numfiles}
echo "finished stage_in.sh"
echo "starting run_mascarpone.sh"
jsrun -n ${nnodes} -g 6 -c 42 -a ${nprocspn} --bind=proportional-packed:7 --launch_distribution=packed stdbuf -o0 ./run_mascarpone.sh |& tee out.${LSB_JOBID}
echo "finished run_mascarpone.sh"
