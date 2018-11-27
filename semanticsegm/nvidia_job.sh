#!/bin/bash
#SBATCH -N 50
#SBATCH -C knl
#SBATCH -t 10:00:00
#SBATCH -A dasrepo
#SBATCH -q premium
#SBATCH --mail-user=amahesh@lbl.gov

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


#run the application:
module load python/2.7-anaconda-4.4
source activate createlabels
srun -n 3400 -c 4 --cpu_bind=cores python -u create_multichannel_cropped_labels.py
