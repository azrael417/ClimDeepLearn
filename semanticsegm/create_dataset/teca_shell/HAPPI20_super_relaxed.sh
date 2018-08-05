#!/bin/bash -l

#SBATCH -A dasrepo
#SBATCH --job-name=teca_HAPPI_20_super_relaxed
#SBATCH --time=02:30:00
#SBATCH --nodes=54
#SBATCH --exclusive
#SBATCH --output=teca_HAPPI_20_super_relaxed.out
#SBATCH --error=teca_HAPPI_20_super_relaxed.err
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

cd /global/cscratch1/sd/mwehner/machine_learning_climate_data/HAPPI20/fvCAM5_HAPPI20_run1/h2/
pwd

files_regex='.*\.nc$'


srun -n 3650 --cpu_bind=cores --mem_bind=local teca_tc_detect \
    --input_regex ${files_regex} \
    --candidate_file /global/project/projectdirs/dasrepo/gb2018/teca/teca_HAPPI20_run1_super_relaxed/candidates.bin \
    --track_file /global/project/projectdirs/dasrepo/gb2018/teca/teca_HAPPI20_run1_super_relaxed/tracks.bin \
    --candidates::min_vorticity_850mb 0.1e-4 \
    --candidates::max_core_temperature_delta 0.1 \
    --candidates::max_pressure_delta 100.0\
