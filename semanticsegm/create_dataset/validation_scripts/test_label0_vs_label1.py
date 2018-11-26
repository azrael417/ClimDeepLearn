import argparse
import numpy as np
import pandas as pd
import h5py as h5
from glob import glob
import os


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, help='select from the following: HAPPI20, HAPPI15, All-Hist')
	parser.add_argument('-y', '--year', type=str, help='The year in which to search for different numbers of TC vs AR classes.')
	parser.add_argument('--run_num', type=str, help='The run number in which to search')
	cli_args = parser.parse_args()

 	label_0_regex = "/global/cscratch1/sd/amahesh/gb_helper/{}/label_0/subtables/*{}*{}.csv".format(cli_args.dataset, 
 		cli_args.year, cli_args.run_num)
 	label_1_regex = "/global/cscratch1/sd/amahesh/gb_helper/{}/label_1/subtables/*{}*{}.csv".format(cli_args.dataset,
 		cli_args.year, cli_args.run_num)

 	label_0_files = sorted(glob(label_0_regex))
 	label_1_files = sorted(glob(label_1_regex))

 	table_name = None

 	for i, label_0_file in enumerate(label_0_files):
 		label_1_file = label_1_files[i]
 		assert os.path.basename(label_0_file) == os.path.basename(label_1_file),\
 			"File names do not match!"

 		label_0_table = pd.read_csv(label_0_file)
 		label_1_table = pd.read_csv(label_1_file)

 		if len(label_0_table) != len(label_1_table):
 			print("Length of Label_0_table: {}, length of label_1_table: {}".format(len(label_0_table), len(label_1_table)))
 			table_name = os.path.basename(label_0_file)
 			break

 	if table_name != None:
		year = int(table_name[12:16])
		month = int(table_name[17:19])
		day = int(table_name[20:22])
		run_num = int(table_name[-5:-4])

		for time_step_index in range(8):
			f = h5.File("/global/cscratch1/sd/amahesh/gb_data/{}/data-{:04d}-{:02d}-{:02d}-{:02d}-{:01d}.h5".format(
				cli_args.dataset, year, month, day, time_step_index, run_num))
			if f['climate']['labels_0_stats'][5][0] < f['climate']['labels_1_stats'][5][0]:
				print("All good! {:04d}-{:02d}-{:02d}-{:02d}-{:01d} has different TC classe proportions for labels_0 and labels_1".format(
					year, month, day, time_step_index, run_num))
				print("Label_0 TC pixels: {}".format(f['climate']['labels_0_stats'][5][0] * 768 * 1152))
				print("Label_1 TC pixels: {}".format(f['climate']['labels_1_stats'][5][0] * 768 * 1152))
				break
	else:
		print("All the subtables are the same length.  There was an error with the subtable creation.")


