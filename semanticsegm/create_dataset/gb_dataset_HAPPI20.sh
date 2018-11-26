#!/bin/bash -l

#SBATCH -A dasrepo
#SBATCH --job-name=gb_HAPPI20
#SBATCH --time=04:30:00
#SBATCH --nodes=22
#SBATCH --exclusive
#SBATCH --output=gb_HAPPI20.out
#SBATCH --error=gb_HAPPI20.err
##SBATCH -p debug 
#SBATCH -C knl,quad,cache
#SBATCH --mail-user=amahesh@lbl.gov
#SBATCH --qos premium


module load python/2.7-anaconda-4.4
source activate createlabels

cd /global/homes/a/amahesh/ClimDeepLearn/semanticsegm/create_dataset
pwd


srun -n 1200 -c 4 --cpu_bind=cores python create_multichannel_multithreshold_labels.py --dataset HAPPI20 \
	--label_output_dir /global/cscratch1/sd/amahesh/gb_data/HAPPI20/ \
	--vis_output_dir /global/cscratch1/sd/amahesh/gb_helper/HAPPI20/image_dump/ --parallel
