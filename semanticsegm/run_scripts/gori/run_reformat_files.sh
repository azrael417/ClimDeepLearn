#!/bin/bash
#SBATCH -C knl
#SBATCH -q regular
#SBATCH -t 02:00:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#load python
module load python/3.6-anaconda-4.4

#run the application:
procs_per_node=32
procs_total=$(( ${procs_per_node} * ${SLURM_NNODES} ))
cores_per_proc=$(( 256 / ${procs_per_node} ))

#paths to files
ensembles="All-Hist HAPPI15 HAPPI20"
#ensembles="HAPPI20"
input_dir=/global/cscratch1/sd/amahesh/gb_data
output_dir=/global/cscratch1/sd/tkurth/gb2018/tiramisu/segm_h5_v3_new

for ensemble in ${ensembles}; do
    srun -n ${procs_total} -c ${cores_per_proc} --cpu_bind=cores python ../../utils/reformat_files.py \
	                        --input_path=${input_dir}/${ensemble} --output_path=${output_dir}/${ensemble} --update
done
