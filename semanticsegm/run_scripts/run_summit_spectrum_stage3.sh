#!/bin/bash
##BSUB -nnodes 512
#BSUB -csm y
#BSUB -R "1*{select[LN]span[hosts=1]} + 43008*{select[CN &&  (hname !='c35n15') && (hname !='d04n05')]order[!-slots:maxslots]span[ptile=42] }"
#BSUB -W 60
#BSUB -P VEN101SUMMIT
#BSUB -alloc_flags "smt4 nvme"
#BSUB -J climseg_training
#BSUB -o out_sc.%J
#BSUB -e out_sc.%J
#BSUB -q tested

#determine number of nodes and total procs
nprocspn=6
nnodes=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)
nprocs=$(( ${nnodes} * ${nprocspn} ))


#script in place
SWORK=/gpfs/alpinetds/scratch/mfatica/ven101
run_dir=${SWORK}/SC_scaling_r2/run_nn${nnodes}_np${nprocs}_j${LSB_JOBID}
#run_dir=${SWORK}/testing/run_nn${nnodes}_np${nprocs}_j${LSB_JOBID}
mkdir -p ${run_dir}

#cp stage_in_2_spectrum.sh ${run_dir}/
cp stage_in.sh ${run_dir}/
cp stage_in_3.sh ${run_dir}/
#cp run_mascarpone.sh ${run_dir}/
cp run_mascarpone_2.sh ${run_dir}/
cp run_mascarpone_2_fp16.sh ${run_dir}/
cp ../tiramisu-tf/stagein.py ${run_dir}/
cp ../tiramisu-tf/stagein_3.py ${run_dir}/
cp ../tiramisu-tf/mascarpone-tiramisu-tf*.py ${run_dir}/
cp ../tiramisu-tf/tiramisu_helpers.py ${run_dir}/

#step in
cd ${run_dir}

#datadir
datadir="/gpfs/alpinetds/world-shared/csc275/climdata_reformat"
#datadir="/gpfs/alpinetds/scratch/mfatica/ven101"
scratchdir="/xfs/scratch/"$(whoami)"/data"
nperproc=250
numfiles=$(( ${nprocspn} * ${nperproc} )) 

#run
cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch > host_list
#jsrun -n ${nnodes} -g 1 -c 42 -a 1 ./stage_in_2_spectrum.sh ${datadir} ${scratchdir} ${numfiles}

echo "starting stage_in.sh"
#jsrun -n ${nnodes} -g 1 -c 42 -a 1 ./stage_in.sh ${datadir} ${scratchdir} ${numfiles}
jsrun -n ${nnodes} -g 1 -c 42 -a 1 ./stage_in_3.sh ${datadir} ${scratchdir} ${numfiles}
echo "finished stage_in.sh"
echo "starting run_mascarpone.sh"
jsrun -n ${nnodes} -g 6 -c 42 -a ${nprocspn} --bind=proportional-packed:7 --launch_distribution=packed stdbuf -o0 ./run_mascarpone_2.sh |& tee out.fp32.${LSB_JOBID}
jsrun -n ${nnodes} -g 6 -c 42 -a ${nprocspn} --bind=proportional-packed:7 --launch_distribution=packed stdbuf -o0 ./run_mascarpone_2_fp16.sh |& tee out.fp16.${LSB_JOBID}
echo "finished run_mascarpone.sh"
