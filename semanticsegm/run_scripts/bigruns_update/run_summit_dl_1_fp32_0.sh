#!/bin/bash
##BSUB -nnodes 1
#BSUB -W 30
#BSUB -P CSC275PRABHAT
##BSUB -P VEN101
#BSUB -alloc_flags "smt4 nvme"
#BSUB -J climseg_training
#BSUB -o out_test.%J
#BSUB -e out_test.%J
#BSUB -q batch
#BSUB -csm y
#BSUB -R "1*{select[LN]span[hosts=1]} + 42*{select[CN && (hname != 'e28n10') && (hname != 'b35n11') && (hname != 'b08n02') && (hname != 'e01n04') && (hname != 'b18n09') && (hname != 'a09n18') && (hname != 'g14n17') && (hname != 'f14n05') && (hname != 'g30n16') && (hname != 'd21n17') && (hname != 'a36n05') && (hname != 'a11n12') && (hname != 'a15n10') && (hname != 'b01n06') && (hname != 'b25n09') && (hname != 'b28n12') && (hname != 'b35n01') && (hname != 'b35n04') && (hname != 'b36n08') && (hname != 'c02n13') && (hname != 'c13n05') && (hname != 'c25n18') && (hname != 'c27n06') && (hname != 'c28n10') && (hname != 'c30n01') && (hname != 'c34n10') && (hname != 'c34n11') && (hname != 'c34n12') && (hname != 'c34n13') && (hname != 'c34n14') && (hname != 'c34n15') && (hname != 'c34n16') && (hname != 'c34n17') && (hname != 'c35n15') && (hname != 'd02n06') && (hname != 'd03n12') && (hname != 'd04n01') && (hname != 'd14n13') && (hname != 'd15n14') && (hname != 'd25n03') && (hname != 'd25n14') && (hname != 'e04n01') && (hname != 'e06n07') && (hname != 'e09n06') && (hname != 'e11n18') && (hname != 'e12n03') && (hname != 'e12n14') && (hname != 'e13n03') && (hname != 'e25n13') && (hname != 'e26n06') && (hname != 'e27n13') && (hname != 'e28n06') && (hname != 'e29n07') && (hname != 'e29n10') && (hname != 'e30n04') && (hname != 'e33n16') && (hname != 'f01n01') && (hname != 'f05n08') && (hname != 'f13n16') && (hname != 'f17n12') && (hname != 'f19n01') && (hname != 'f23n02') && (hname != 'f24n09') && (hname != 'f29n17') && (hname != 'f32n06') && (hname != 'f32n15') && (hname != 'g02n11') && (hname != 'g18n09') && (hname != 'g21n17') && (hname != 'g23n08') && (hname != 'g23n10') && (hname != 'h19n07') && (hname != 'h33n01') && (hname != 'h34n04') && (hname != 'f28n03') && (hname != 'g16n04') && (hname != 'f04n12') && (hname != 'e03n01') && (hname != 'e30n13') && (hname != 'h29n06') && (hname != 'c08n16') && (hname != 'c08n15') && (hname != 'a23n03') && (hname != 'h29n06') && (hname != 'f26n03') && (hname != 'g13n05') && (hname != 'b25n17') && (hname != 'b36n17') && (hname != 'd27n01') && (hname != 'd03n01') && (hname != 'f18n14') && (hname != 'f17n01') && (hname != 'f17n09') && (hname != 'f06n09') && (hname != 'f12n12') && (hname != 'd06n02') && (hname != 'd28n04') && (hname != 'c35n09') && (hname != 'd21n16') && (hname != 'f19n14') && (hname != 'd21n16') && (hname != 'b08n02')]order[!-slots:maxslots]span[ptile=42] }"

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
cp run_deeplab.sh ${run_dir}/
cp run_deeplab_fp16.sh ${run_dir}/
cp ../../deeplab-tf/*.py ${run_dir}/

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

echo "starting run_deeplab.sh" `date`
# NOTE: arguments after scratchdir are (epochs, lr, scale_factor, gradient-lag)

#fp32-lag0
jsrun  -n ${nnodes} -g 6 -c 42 -a ${nprocspn} --bind=proportional-packed:7 --launch_distribution=packed stdbuf -o0 ./run_deeplab.sh ${scratchdir} 10 0.0001 0.1 0 1 |& tee out.fp32.lag0.${LSB_JOBID}
echo "finished run_mascarpone.sh" `date`
