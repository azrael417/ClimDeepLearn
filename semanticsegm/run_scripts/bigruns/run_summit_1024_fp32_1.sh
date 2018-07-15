#!/bin/bash
#BSUB -csm y
#BSUB -R "1*{select[LN]span[hosts=1]} + 43008*{select[CN && (hname != 'e08n05') && (hname != 'a07n09') && (hname != 'a15n10') && (hname != 'b21n06') && (hname != 'b25n09') && (hname != 'b28n12') && (hname != 'b36n08') && (hname != 'c01n01') && (hname != 'c01n02') && (hname != 'c01n03') && (hname != 'c01n04') && (hname != 'c01n05') && (hname != 'c01n06') && (hname != 'c01n07') && (hname != 'c01n08') && (hname != 'c01n09') && (hname != 'c01n10') && (hname != 'c01n11') && (hname != 'c01n12') && (hname != 'c01n13') && (hname != 'c01n14') && (hname != 'c01n15') && (hname != 'c01n16') && (hname != 'c01n17') && (hname != 'c01n18') && (hname != 'c04n05') && (hname != 'c13n05') && (hname != 'c27n06') && (hname != 'c28n10') && (hname != 'c35n15') && (hname != 'd02n06') && (hname != 'd15n14') && (hname != 'd21n16') && (hname != 'e04n01') && (hname != 'e06n07') && (hname != 'e11n18') && (hname != 'e13n03') && (hname != 'e26n06') && (hname != 'e29n10') && (hname != 'e33n16') && (hname != 'e34n06') && (hname != 'f04n02') && (hname != 'f05n08') && (hname != 'f31n10') && (hname != 'f31n11') && (hname != 'f31n12') && (hname != 'f31n16') && (hname != 'f32n01') && (hname != 'f32n03') && (hname != 'f32n06') && (hname != 'f32n12') && (hname != 'f32n14') && (hname != 'f32n16') && (hname != 'f32n18') && (hname != 'f33n08') && (hname != 'f33n10') && (hname != 'f33n15') && (hname != 'f33n16') && (hname != 'f33n17') && (hname != 'f34n02') && (hname != 'f35n06') && (hname != 'f35n07') && (hname != 'f36n02') && (hname != 'f36n03') && (hname != 'f36n08') && (hname != 'f36n11') && (hname != 'h09n04') && (hname != 'h10n05') && (hname != 'h10n06') && (hname != 'h10n07') && (hname != 'h10n09') && (hname != 'h10n11') && (hname != 'h10n15') && (hname != 'h10n17') && (hname != 'h11n01') && (hname != 'h11n02') && (hname != 'h11n08') && (hname != 'h11n16') && (hname != 'h12n02') && (hname != 'h12n18') && (hname != 'h14n06') && (hname != 'h14n10') && (hname != 'h14n12') && (hname != 'h14n17') && (hname != 'h15n02') && (hname != 'h15n03') && (hname != 'h15n04') && (hname != 'h15n05') && (hname != 'h15n06') && (hname != 'h15n09') && (hname != 'h15n14') && (hname != 'h16n02') && (hname != 'h16n07') && (hname != 'h16n14') && (hname != 'h16n16') && (hname != 'h17n02') && (hname != 'h17n04') && (hname != 'h17n09') && (hname != 'h17n13') && (hname != 'h17n16') && (hname != 'h17n17') && (hname != 'h17n18') && (hname != 'h18n01') && (hname != 'h18n09') && (hname != 'h18n16') && (hname != 'h19n02') && (hname != 'h19n03') && (hname != 'h19n08') && (hname != 'e28n10') && (hname != 'e29n04') && (hname != 'e29n06') && (hname != 'e29n11') && (hname != 'e30n07') && (hname != 'e32n08') && (hname != 'e32n12') && (hname != 'e33n02') && (hname != 'e33n12') && (hname != 'e33n15') && (hname != 'e34n12') && (hname != 'e34n18') && (hname != 'e35n04') && (hname != 'e36n01') && (hname != 'e36n11') && (hname != 'e36n12') && (hname != 'e36n14') && (hname != 'f01n07') && (hname != 'f01n17') && (hname != 'f01n18') && (hname != 'f02n07') && (hname != 'f02n09') && (hname != 'f02n17') && (hname != 'f03n11') && (hname != 'f03n16') && (hname != 'f04n13') && (hname != 'f04n16') && (hname != 'f05n01') && (hname != 'f05n04') && (hname != 'f05n12') && (hname != 'f05n13') && (hname != 'f05n17') && (hname != 'f06n02') && (hname != 'f06n07') && (hname != 'f06n17') && (hname != 'f06n18') && (hname != 'f07n05') && (hname != 'f07n06') && (hname != 'e08n11')]order[!-slots:maxslots]span[ptile=42] }"

#BSUB -W 60
##BSUB -P CSC275PRABHAT
#BSUB -P VEN101SUMMIT
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
SWORK=/gpfs/alpinetds/world-shared/ven201/seant/climate/gb_runs
script_dir=`pwd`
run_dir=${SWORK}/GB_solution_final/run_nn${nnodes}_np${nprocs}_j${LSB_JOBID}_1
mkdir -p ${run_dir}

cp ${script_dir}/stage_in_parallel.sh ${run_dir}/
cp ${script_dir}/run_mascarpone.sh ${run_dir}/
cp ${script_dir}/run_mascarpone_fp16.sh ${run_dir}/
cp ${script_dir}/../../tiramisu-tf/parallel_stagein.py ${run_dir}/
cp ${script_dir}/../../tiramisu-tf/mascarpone-tiramisu-tf*.py ${run_dir}/
cp ${script_dir}/../../tiramisu-tf/tiramisu_helpers.py ${run_dir}/

#step in
cd ${run_dir}

#datadir
datadir="/gpfs/alpinetds/world-shared/csc275/climdata_reformat"
scratchdir="/xfs/scratch/"$(whoami)""
nperproc=250
numfiles=$(( ${nprocspn} * ${nperproc} )) 

#run
cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch > host_list

#xfs health
jsrun -n ${nnodes} -g 1 -c 42 -a 1 df ${scratchdir}

echo "starting stage_in_parallel.sh " `date`
jsrun -n ${nnodes} -g 1 -c 42 -a 1 ./stage_in_parallel.sh ${datadir} ${scratchdir} ${numfiles}
echo "finished stage_in_parallel.sh" `date`

# Set flag after stage-in to prepend to spectrum-mpi's existing LD_PRELOAD
export OMPI_LD_PRELOAD_PREPEND="${scratchdir}/pyvenv_summit_v4/lib/directconv.so"

echo "starting run_mascarpone.sh" `date`
# NOTE: arguments after scratchdir are (epochs, lr, scale_factor, gradient-lag)

#fp32-lag1 
jsrun  -n ${nnodes} -g 6 -c 42 -a ${nprocspn} --bind=proportional-packed:7 --launch_distribution=packed stdbuf -o0 ./run_mascarpone.sh ${scratchdir} 25 0.4096 0.1 1 |& tee out.fp32.lag1.${LSB_JOBID}
echo "finished run_mascarpone.sh" `date`
#----
