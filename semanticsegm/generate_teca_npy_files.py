from teca import *
import os
import pickle
from glob import glob
import csv
import argparse

# filenames = ['/global/cscratch1/sd/mwehner/machine_learning_climate_data/HAPPI20/fvCAM5_HAPPI20_run1/TECA2/wind_tracks_fvCAM5_HAPPI20_run1.bin',
#              '/global/cscratch1/sd/mwehner/machine_learning_climate_data/HAPPI20/fvCAM5_HAPPI20_run3/TECA2/wind_tracks_fvCAM5_HAPPI20_run3.bin',
#              '/global/cscratch1/sd/mwehner/machine_learning_climate_data/HAPPI20/fvCAM5_HAPPI20_run4/TECA2/wind_tracks_fvCAM5_HAPPI20_run4.bin',
#              '/global/cscratch1/sd/mwehner/machine_learning_climate_data/HAPPI20/fvCAM5_HAPPI20_run2/TECA2/wind_tracks_fvCAM5_HAPPI20_run2.bin',
#              '/global/cscratch1/sd/mwehner/machine_learning_climate_data/HAPPI20/fvCAM5_HAPPI20_run6/TECA2/wind_tracks_fvCAM5_HAPPI20_run6.bin']

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, help='select from the following: HAPPI20, HAPPI15, All-Hist')
	parser.add_argument('--output_dir', type=str, help='where to output the numpy files')
	parser.add_argument('--label_version', type=str, help='label_0, label_1, or label_2')
	cli_args = parser.parse_args()

	if cli_args.label_version == 'label_0':
		if cli_args.dataset == 'All-Hist':
			teca_regex = '/global/cscratch1/sd/mwehner/machine_learning_climate_data/All-Hist/CAM5-1-0.25degree_All-Hist_est1_v3_run*/TECA2/wind_tracks_CAM5-1-0.25degree_All-Hist_est1_v3_run*.bin'
		elif cli_args.dataset == 'HAPPI20':
			teca_regex = '/global/cscratch1/sd/mwehner/machine_learning_climate_data/HAPPI20/fvCAM5_HAPPI20_run[12346]/TECA2/wind_tracks_fvCAM5_HAPPI20_run[12346].bin'
		elif cli_args.dataset == 'HAPPI15':
			teca_regex = '/global/cscratch1/sd/mwehner/machine_learning_climate_data/HAPPI15/fvCAM5_HAPPI15_run*/TECA2/wind_tracks_fvCAM5_HAPPI15_run*.bin'
	elif cli_args.label_version == 'label_1':
		teca_regex = "/global/project/projectdirs/dasrepo/gb2018/teca/teca_{}_run*_super_relaxed/wind_tracks.bin".format(cli_args.dataset)

filenames = glob(teca_regex)
happi_20_label_0_run_5 = "/global/project/projectdirs/dasrepo/gb2018/teca/teca_HAPPI20_run5_default/wind_tracks.bin"
if cli_args.dataset == "HAPPI20" and cli_args.label_version == 'label_0':
	filenames.append(happi_20_label_0_run_5)

print("Number of files found: {}".format(len(filenames)))
#filename = '/global/cscratch1/sd/karthik_/TECA2.0Demo/demo_tracks/wind_tracks_CAM5-1-0.25degree_All-Hist_est1_v3_run2.bin'
#filename = './candidates_CAM5-1-0.25degree_All-Hist_est1_v3_run2.bin'
#filename = './tables_CAM5-1-0.25degree_All-Hist_est1_v3_run2.bin'
track_id = []
lon = []
lat = []
r0 = []
wind = []
time = []
year = []
month = []
day = []
hour = []
storm_id = []
run_nums = []


for filename in filenames:
	# load the full set of tables
	reader = teca_table_reader.New()
	reader.set_file_name(filename)

	tap_1 = teca_dataset_capture.New()
	tap_1.set_input_connection(reader.get_output_port())
	tap_1.update()

	table = as_teca_table(tap_1.get_dataset())

	# print some info
	print 'In tables file: %s'%(filename)
	print 'Found Columns:'
	# for i in xrange(table.get_number_of_columns()):
	#   print '%i - %s'%(i, table.get_column_name(i))
	# print 'Number of rows: %d'%(table.get_number_of_rows())

	track_id.extend(list(table.get_column('track_id').as_array()))
	lon.extend(list(table.get_column('lon').as_array()))
	lat.extend(list(table.get_column('lat').as_array()))
	r0.extend(list(table.get_column('wind_radius_0').as_array()))
	wind.extend(list(table.get_column('surface_wind').as_array()))
	time.extend(list(table.get_column('time').as_array()))
	year.extend(list(table.get_column('year').as_array()))
	month.extend(list(table.get_column('month').as_array()))
	day.extend(list(table.get_column('day').as_array()))
	hour.extend(list(table.get_column('hour').as_array()))
	storm_id.extend(list(table.get_column('storm_id').as_array()))

	if filename == happi_20_label_0_run_5:
		run_nums.extend([5] * table.get_number_of_rows())
	else:
		if cli_args.dataset == 'label_0':
			run_nums.extend([int(filename[-5:-4])] * table.get_number_of_rows())
		else:
			index_of_run = filename.find('run')
			run_nums.extend([int(filename[index_of_run+3])] * table.get_number_of_rows())
	# print(len(track_id))
	#print(lon.size)
	#print(lat.size)
	#print(r0.size)
	#print(wind.size)
	#print(time.size)
	#print(year.size)
	#print(month.size)
	#print(day.size)
	#print(hour.size)
	# print(len(storm_id))

	print("num_rows: {}, run_num: {}".format(table.get_number_of_rows(), filename[index_of_run+3]))

save_filepath = cli_args.output_dir

np.save(save_filepath+"track_id.npy", np.asarray(track_id))
np.save(save_filepath+"lon.npy", np.asarray(lon))
np.save(save_filepath+"lat.npy", np.asarray(lat))
np.save(save_filepath+"r0.npy", np.asarray(r0))
np.save(save_filepath+"wind.npy", np.asarray(wind))
np.save(save_filepath+"time.npy", np.asarray(time))
np.save(save_filepath+"year.npy", np.asarray(year))
np.save(save_filepath+"month.npy", np.asarray(month))
np.save(save_filepath+"day.npy", np.asarray(day))
np.save(save_filepath+"hour.npy", np.asarray(hour))
np.save(save_filepath+"storm_id.npy", np.asarray(storm_id))
np.save(save_filepath+"run_num.npy", np.asarray(run_nums))