#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH --mail-user=amahesh@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -p regular
#SBATCH -t 09:30:00
#SBATCH -L SCRATCH

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


#run the application:
module load python/2.7-anaconda-4.4
source activate createlabels
srun -n 30 -c 2 --cpu_bind=cores python create_combined_cropped_labels.py
