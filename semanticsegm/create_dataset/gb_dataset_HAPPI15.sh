#!/bin/bash -l

#SBATCH -A dasrepo
#SBATCH --job-name=gb_HAPPI15
#SBATCH --time=01:45:00
#SBATCH --nodes=64
#SBATCH --exclusive
#SBATCH --output=gb_HAPPI15.out
#SBATCH --error=gb_HAPPI15.err
##SBATCH -p debug 
#SBATCH -C knl,quad,cache
#SBATCH --mail-user=amahesh@lbl.gov
#SBATCH --qos premium


module load python/2.7-anaconda-4.4
source activate createlabels

cd /global/homes/a/amahesh/ClimDeepLearn/semanticsegm/create_dataset
pwd


srun -n 4352 --cpu_bind=cores python create_multichannel_multithreshold_labels.py --dataset HAPPI15 \
	--label_output_dir /global/cscratch1/sd/amahesh/gb_data/HAPPI15/ \
	--vis_output_dir /global/cscratch1/sd/amahesh/gb_helper/HAPPI15/image_dump/
