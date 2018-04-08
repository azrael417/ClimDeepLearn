#!/bin/bash
#BSUB -csm y
#BSUB -R "1*{select[LN]span[hosts=1]} + 10752*{select[CN && (hname != 'a07n09') && (hname != 'a15n10') && (hname != 'b21n06') && (hname != 'b25n09') && (hname != 'b28n12') && (hname != 'b36n08') && (hname != 'c01n01') && (hname != 'c01n02') && (hname != 'c01n03') && (hname != 'c01n04') && (hname != 'c01n05') && (hname != 'c01n06') && (hname != 'c01n07') && (hname != 'c01n08') && (hname != 'c01n09') && (hname != 'c01n10') && (hname != 'c01n11') && (hname != 'c01n12') && (hname != 'c01n13') && (hname != 'c01n14') && (hname != 'c01n15') && (hname != 'c01n16') && (hname != 'c01n17') && (hname != 'c01n18') && (hname != 'c04n05') && (hname != 'c13n05') && (hname != 'c27n06') && (hname != 'c28n10') && (hname != 'c35n15') && (hname != 'd02n06') && (hname != 'd15n14') && (hname != 'd21n16') && (hname != 'e04n01') && (hname != 'e06n07') && (hname != 'e11n18') && (hname != 'e13n03') && (hname != 'e26n06') && (hname != 'e29n10') && (hname != 'e33n16') && (hname != 'e34n06') && (hname != 'f04n02') && (hname != 'f05n08')]order[!-slots:maxslots]span[ptile=42] }"
#BSUB -W 120
#BSUB -P CSC275PRABHAT
##BSUB -P VEN101
#BSUB -alloc_flags "smt4 nvme"
#BSUB -J climseg_training
#BSUB -o out_test.%J
#BSUB -e out_test.%J
#BSUB -q new

# load modules
module load cuda


#determine number of nodes and total procs
nprocspn=6
nnodes=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)
nprocs=$(( ${nnodes} * ${nprocspn} ))

#script in place
#SWORK=/gpfs/alpinetds/scratch/mfatica/ven101/
run_dir=${SWORK}/GB_scale_fp32_v2/run_nn${nnodes}_np${nprocs}_j${LSB_JOBID}
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

# Set flag after stage-in to prepend to spectrum-mpi's existing LD_PRELOAD
#export OMPI_LD_PRELOAD_PREPEND=/gpfs/alpinetds/world-shared/ven201/seant/climate/pyvenv_summit_v3/lib/directconv.so
export OMPI_LD_PRELOAD_PREPEND="${scratchdir}/pyvenv_summit_v4/lib/directconv.so"

echo "starting run_mascarpone.sh"
# NOTE: arguments after scratchdir are (epochs, lr, scale_factor, gradient-lag)
jsrun  -n ${nnodes} -g 6 -c 42 -a ${nprocspn} --bind=proportional-packed:7 --launch_distribution=packed stdbuf -o0 ./run_mascarpone.sh ${scratchdir} 5 0.0001 0.1 0 |& tee out.fp32.${LSB_JOBID}

#fp16
#jsrun  -n ${nnodes} -g 6 -c 42 -a ${nprocspn} --bind=proportional-packed:7 --launch_distribution=packed stdbuf -o0 ./run_mascarpone_fp16.sh ${scratchdir} 2 0.0001 0.1 0 |& tee out.fp16.${LSB_JOBID}


echo "finished run_mascarpone.sh"
