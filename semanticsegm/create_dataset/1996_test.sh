#!/bin/bash -l

#SBATCH -A dasrepo
#SBATCH --job-name=199_24_test
#SBATCH --time=0:30:00
#SBATCH --nodes=6
#SBATCH --output=/global/homes/a/amahesh/ClimDeepLearn/semanticsegm/create_dataset/1996_TMQ_width_area_thresh_24.out
#SBATCH --error=/global/homes/a/amahesh/ClimDeepLearn/semanticsegm/create_dataset/1996_TMQ_width_area_thresh_24.err
#SBATCH -p debug 
#SBATCH -C knl,quad,cache
#SBATCH --mail-user=amahesh@lbl.gov
##SBATCH --qos premium

module load python/2.7-anaconda-4.4
source activate createlabels
export PYTHONPATH=$PYTHONPATH:~/ClimDeepLearn/semanticsegm/

srun -n 366 --cpu_bind=cores python /global/homes/a/amahesh/ClimDeepLearn/semanticsegm/create_dataset/create_multichannel_cropped_labels.py --dataset All-Hist --label_output_dir /global/cscratch1/sd/amahesh/1996_gb_tests/TMQ_width_area_thresh_24/ --vis_output_dir /global/cscratch1/sd/amahesh/1996_gb_tests/TMQ_width_area_thresh_24/image_dump/ --ar_threshold TMQ --parallel
