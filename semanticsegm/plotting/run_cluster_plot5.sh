#!/bin/bash
#SBATCH -N 1
#SBATCH -C knl
#SBATCH -A dasrepo
##SBATCH -p debug
#SBATCH --qos premium
#SBATCH -J makeclimplot1 
#SBATCH --mail-user=mudigonda@berkeley.edu
#SBATCH --output=run_plot1_output.out
#SBATCH --error=run_plot1_error.err
#SBATCH --mail-type=ALL
#SBATCH -t 10:00:00
#SBATCH -L SCRATCH
#SBATCH --exclusive

module load python/2.7-anaconda-4.4
source activate climDL

#run the application:
srun -n 1 -c 64 --cpu_bind=cores /global/homes/m/mayur/Projects/git_climdeeplearn/ClimDeepLearn/semanticsegm/plotting/run_plot5.sh 
