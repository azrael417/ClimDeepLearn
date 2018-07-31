#!/bin/bash -l

##SBATCH -A dasrepo
#SBATCH --job-name=Low2
#SBATCH --time=0:30:00
#SBATCH --nodes=10
#SBATCH --exclusive
#SBATCH --output=teca%jodid.out
#SBATCH --error=teca%jodid.err
#SBATCH -p debug 
#SBATCH -C knl,quad,cache
##SBATCH --qos premium

module unload nco/4.6.0
module unload nco
module unload cdo
module unload cray-netcdf

module load teca/2.1.1-knl

cd /global/cscratch1/sd/mwehner/machine_learning_climate_data/All-Hist/CAM5-1-0.25degree_All-Hist_est1_v3_run1/h2/
pwd

files_regex='CAM5-1-0.25degree_All-Hist_est1_v3_run1.cam.h2.2001*.nc'

srun -n 680 --cpu_bind=cores --mem_bind=local teca_tc_detect                     \
    --input_regex ${files_regex}    \
    --candidate_file /global/homes/m/mayur/tmp/candidates_CAM5-1-0.25degree_Low_Aero_run2.bin        \
    --track_file /global/homes/m/mayur/tmp/tracks_CAM5-1-0.25degree_Low_Aero_run2.bin \
    --candidates::min_vorticity_850mb 1.6e-4 \
    --candidates::max_core_temperature_delta 0.8 \
    --candidates::max_pressure_delta 400.0 
