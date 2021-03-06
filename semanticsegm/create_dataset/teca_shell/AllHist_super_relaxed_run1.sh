#!/bin/bash -l

#SBATCH -A dasrepo
#SBATCH --job-name=teca_AllHist_super_relaxed
#SBATCH --time=04:30:00
#SBATCH --nodes=108
#SBATCH --exclusive
#SBATCH --output=teca_All-Hist_super_relaxed.out
#SBATCH --error=teca_All-Hist_super_relaxed.err
##SBATCH -p debug 
#SBATCH -C knl,quad,cache
#SBATCH --mail-user=amahesh@lbl.gov
#SBATCH --qos premium

module unload nco/4.6.0
module unload nco
module unload cdo
module unload cray-netcdf

module unload python/2.7-anaconda-4.4
module load teca/2.1.1-knl

cd /global/cscratch1/sd/mwehner/machine_learning_climate_data/All-Hist/CAM5-1-0.25degree_All-Hist_est1_v3_run1/h2/
pwd

files_regex='.*\.nc$'


srun -n 7300 --cpu_bind=cores --mem_bind=local teca_tc_detect \
    --input_regex ${files_regex} \
    --candidate_file /global/project/projectdirs/dasrepo/gb2018/teca/teca_All-Hist_run1_super_relaxed/candidates.bin \
    --track_file /global/project/projectdirs/dasrepo/gb2018/teca/teca_All-Hist_run1_super_relaxed/tracks.bin \
    --candidates::min_vorticity_850mb 0.1e-4 \
    --candidates::max_core_temperature_delta 0.1 \
    --candidates::max_pressure_delta 100.0\
