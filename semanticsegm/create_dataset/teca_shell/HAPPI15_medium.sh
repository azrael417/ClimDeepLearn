#!/bin/bash -l

#SBATCH -A dasrepo
#SBATCH --job-name=med
#SBATCH --time=1:40:00
#SBATCH --nodes=44
#SBATCH --exclusive
#SBATCH --output=teca_HAPPI15_medium.out
#SBATCH --error=teca_HAPPI15_medium.err
##SBATCH -p debug 
#SBATCH -C knl,quad,cache
#SBATCH --mail-user=mudigonda@lbl.gov
#SBATCH --qos premium

module unload nco/4.6.0
module unload nco
module unload cdo
module unload cray-netcdf

module unload python/2.7-anaconda-4.4
module load teca/2.1.1-knl

cd /global/cscratch1/sd/mwehner/machine_learning_climate_data/HAPPI15/fvCAM5_HAPPI15_run1/h2/
pwd

files_regex='.*2106.*\.nc$'


srun -n 2920 --cpu_bind=cores --mem_bind=local teca_tc_detect \
    --input_regex ${files_regex} \
    --candidate_file /global/project/projectdirs/dasrepo/gb2018/teca/teca_HAPPI15_run1_medium/candidates.bin \
    --track_file /global/project/projectdirs/dasrepo/gb2018/teca/teca_HAPPI15_run1_medium/tracks.bin \ 
    --candidates::min_vorticity_850mb .5e-4 \
    --candidates::max_core_temperature_delta 0.3 \
    --candidates::max_pressure_delta 150.0\
