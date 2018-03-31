#!/bin/bash
##BSUB -csm y
##BSUB -R "1*{select[LN]span[hosts=1]} + 168*{select[CN &&  (hname !='c35n15') && (hname !='d04n05') && (hname != 'b25n01') && (hname != 'b25n02') && (hname != 'b25n03') && (hname != 'b25n04') && (hname != 'b25n05') && (hname != 'b25n06') && (hname != 'b25n07') && (hname != 'b25n08') && (hname != 'b25n09') && (hname != 'b25n10') && (hname != 'b25n11') && (hname != 'b25n12') && (hname != 'b25n13') && (hname != 'b25n14') && (hname != 'b25n15') && (hname != 'b25n16')]order[!-slots:maxslots]span[ptile=42] }"
#BSUB -W 60
#BSUB -P CSC275PRABHAT
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
run_dir=run_nn64_np384_j53333
#run_dir=${SWORK}/tuning_new/run_nn${nnodes}_np${nprocs}_j${LSB_JOBID}
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
scratchdir="/xfs/scratch/"$(whoami)""
nperproc=250
numfiles=$(( ${nprocspn} * ${nperproc} )) 

#run
cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch > host_list

echo "starting stage_in_parallel.sh"
jsrun -n ${nnodes} -g 1 -c 42 -a 1 ./stage_in_parallel.sh ${datadir} ${scratchdir} ${numfiles}
echo "finished stage_in_parallel.sh"

echo "starting run_mascarpone.sh"
jsrun -n ${nnodes} -g 6 -c 42 -a ${nprocspn} --bind=proportional-packed:7 --launch_distribution=packed stdbuf -o0 ./run_mascarpone.sh ${scratchdir} |& tee out.fp32.${LSB_JOBID}
#jsrun -n ${nnodes} -g 6 -c 42 -a ${nprocspn} --bind=proportional-packed:7 --launch_distribution=packed stdbuf -o0 ./run_mascarpone_fp16.sh ${scratchdir} |& tee out.fp16.${LSB_JOBID}
echo "finished run_mascarpone.sh"
