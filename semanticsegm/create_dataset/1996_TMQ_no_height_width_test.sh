#!/bin/bash -l

#SBATCH -A dasrepo
#SBATCH --job-name=1996_TMQ_width_none
#SBATCH --time=0:30:00
#SBATCH --nodes=44
#SBATCH --output=/global/homes/a/amahesh/ClimDeepLearn/semanticsegm/create_dataset/1996_TMQ_width_none_thresh.out
#SBATCH --error=/global/homes/a/amahesh/ClimDeepLearn/semanticsegm/create_dataset/1996_TMQ_width_none_thresh.err
##SBATCH -p debug 
#SBATCH -C knl,quad,cache
#SBATCH --mail-user=amahesh@lbl.gov
#SBATCH --qos premium

module load python/2.7-anaconda-4.4
source activate createlabels
export PYTHONPATH=$PYTHONPATH:~/ClimDeepLearn/semanticsegm/

srun -n 2920 --cpu_bind=cores python /global/homes/a/amahesh/ClimDeepLearn/semanticsegm/create_dataset/create_multichannel_cropped_labels.py --dataset All-Hist --label_output_dir /global/cscratch1/sd/amahesh/1996_gb_tests/TMQ_width_none_thresh/ --vis_output_dir /global/cscratch1/sd/amahesh/1996_gb_tests/TMQ_width_none_thresh/image_dump/ --ar_threshold TMQ --parallel --no_width
