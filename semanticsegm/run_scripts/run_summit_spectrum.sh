#!/bin/bash
#BSUB -nnodes 256
#BSUB -W 60
#BSUB -P VEN101SUMMIT
#BSUB -alloc_flags "smt4 nvme"
#BSUB -J climseg_training
#BSUB -o out.%J
#BSUB -e out.%J
#BSUB -q new
##BSUB -q batch

#determine number of nodes and total procs
nprocspn=6
nnodes=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)
nprocs=$(( ${nnodes} * ${nprocspn} ))

#export PAMI_ENABLE_STRIPING=0
#export PAMI_IBV_ENABLE_DCT=1
#export CUDA_CACHE_PATH=/tmp


#script in place
SWORK=/gpfs/alpinetds/scratch/mfatica/ven101
run_dir=${SWORK}/tuning/run_nn${nnodes}_np${nprocs}_j${LSB_JOBID}
mkdir -p ${run_dir}
cp stage_in_2.sh ${run_dir}/
cp run_mascarpone.sh ${run_dir}/
cp ../tiramisu-tf/stagein.py ${run_dir}/
cp ../tiramisu-tf/mascarpone-tiramisu-tf*.py ${run_dir}/
cp ../tiramisu-tf/tiramisu_helpers.py ${run_dir}/

#step in
cd ${run_dir}

#datadir
#datadir="/gpfs/alpinetds/scratch/tkurth/csc190/segm_h5_v3_reformat"
datadir="/gpfs/alpinetds/world-shared/csc275/climdata_reformat/"
scratchdir="/xfs/scratch/"$(whoami)"/data"

#compute number of stagein files dependent on data volume
numfiles=$(ls ${datadir} | wc -l)
stagecount=$(( ${numfiles} / ${nnodes} ))

#run
cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch > host_list
jsrun -n ${nnodes} -g 1 -c 42 -a 1 ./stage_in_2.sh ${datadir} ${scratchdir}
jsrun -n ${nnodes} -g 6 -c 42 -a 6 --bind=proportional-packed:7 --launch_distribution=packed stdbuf -o0 ./run_mascarpone.sh |& tee out.${LSB_JOBID}

#mpirun -np ${nnodes} --bind-to none -x PATH -x LD_LIBRARY_PATH --hostfile host_list -npernode 1 ./stage_in_2.sh ${datadir} ${scratchdir}
#mpirun -np ${nprocs} --bind-to none -x PATH -x LD_LIBRARY_PATH --hostfile host_list -npernode ${nprocspn} python ./mascarpone-tiramisu-tf-singlefile.py --blocks 3 3 4 7 10 --loss weighted --optimizer "LARC-Adam" --lr 1e-5 --datadir ${scratchdir} |& tee out.${LSB_JOBID}
