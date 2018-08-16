import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
import argparse
import numpy as np
import pandas as pd
import h5py as h5
from glob import glob
import os
import netCDF4 as nc

def get_lats_lons():
	with nc.Dataset("/global/cscratch1/sd/mwehner/machine_learning_climate_data/All-Hist/CAM5-1-0.25degree_All-Hist_est1_v3_run1/h2/CAM5-1-0.25degree_All-Hist_est1_v3_run1.cam.h2.2015-12-31-00000.nc") as fin:
		lats = fin['lat'][:]
		lons = fin['lon'][:]
	return lats, lons

def plot_mask(lons, lats, img_array, storm_mask,
              year, month, day, time_step_index, run_num, print_field=""):
  my_map = Basemap(projection='robin', llcrnrlat=min(lats), lon_0=np.median(lons),
                  llcrnrlon=min(lons), urcrnrlat=max(lats), urcrnrlon=max(lons), resolution = 'c')
 
  xx, yy = np.meshgrid(lons, lats)
  x_map,y_map = my_map(xx,yy)
  #x_plot, y_plot = my_map(storm_lon, storm_lat)
  my_map.drawcoastlines(color="black")
  my_map.contourf(x_map,y_map,img_array,64,cmap='viridis')
  #my_map.plot(x_plot, y_plot, 'r*', color = "red")
  cbar = my_map.colorbar()
  my_map.contourf(x_map,y_map,np.ma.masked_less(storm_mask,0.009), vmin=0, vmax=2, alpha=0.42,cmap='bwr')
  my_map.drawmeridians(np.arange(-180, 180, 60), labels=[0,0,0,1])
  my_map.drawparallels(np.arange(-90, 90, 30), labels =[1,0,0,0])
  plt.title("{:04d}-{:02d}-{:02d}-{:02d} Label {}".format(year,month,day,time_step_index, print_field[-1]))
  cbar.ax.set_ylabel('TMQ kg $m^{-2}$')

  mask_ex = plt.gcf()
  #mask_ex.savefig("/global/cscratch1/sd/amahesh/segm_plots/combined_mask+{:04d}-{:02d}-{:02d}-{:02d}-{:02d}.png".format(year,month,day,time_step_index, run_num))
  mask_ex.savefig("{}{}combined_mask+{:04d}-{:02d}-{:02d}-{:02d}-{:02d}.png".format(vis_output_dir, print_field, year,month,day,time_step_index, run_num))
  plt.clf()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, help='select from the following: HAPPI20, HAPPI15, All-Hist')
	parser.add_argument('-y', '--year', type=str, help='The year in which to search for different numbers of TC vs AR classes.')
	parser.add_argument('--run_num', type=str, help='The run number in which to search')
	parser.add_argument('--vis_output_dir', type=str, help='The directory in which to output the visualizations of the differneces')
	cli_args = parser.parse_args()

	vis_output_dir = cli_args.vis_output_dir or "/global/cscratch1/sd/amahesh/gb_helper/label0_label1_diff/{}/".format(cli_args.dataset)

	lats, lons = get_lats_lons()

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
				
				save_data = f['climate']['data']
				save_label_1 = f['climate']["labels_1"]
				save_label_0 = f['climate']["labels_0"]

				plot_mask(lons, lats, save_data[:,:,0], save_label_0,
					year, month, day, time_step_index, run_num, print_field="label_0")
				plot_mask(lons, lats, save_data[:,:,0], save_label_1,
					year, month, day, time_step_index, run_num, print_field="label_1")
				break
	else:
		print("All the subtables are the same length.  There was an error with the subtable creation.")

