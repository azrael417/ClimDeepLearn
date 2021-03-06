#!/bin/bash -l
#SBATCH -A dasrepo
#SBATCH --job-name=teca_HAPPI_15_wind_track_super_relaxed_1
#SBATCH --output=teca_HAPPI_15_wind_tracks_super_relaxed_run3.out
#SBATCH --error=teca_HAPPI_15_wind_tracks_super_relaxed_run3.err
#SBATCH -p debug
#SBATCH -N 4
#SBATCH -t 00:30:00
#SBATCH -C haswell
##SBATCH -q premium

module unload python/2.7-anaconda-4.4
module load teca
cd /global/cscratch1/sd/mwehner/machine_learning_climate_data/All-Hist/CAM5-1-0.25degree_All-Hist_est1_v3_run3/h2/

files_regex='.*\.nc$'
track_file=/global/project/projectdirs/dasrepo/gb2018/teca/teca_All-Hist_run3_super_relaxed/tracks.bin
track_file_out=/global/project/projectdirs/dasrepo/gb2018/teca/teca_All-Hist_run3_super_relaxed/wind_tracks.bin

srun -n 4 --ntasks-per-node=1 \
    teca_tc_wind_radii --n_threads 32 --first_track 0 \
    --last_track -1 --wind_files ${files_regex} --track_file ${track_file} \
    --track_file_out ${track_file_out}
