import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
import argparse
import h5py as h5
import numpy as np
from glob import glob
import netCDF4 as nc
import os

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
  my_map.contourf(x_map,y_map,storm_mask, alpha=0.42,cmap='gray')
  my_map.drawmeridians(np.arange(-180, 180, 60), labels=[0,0,0,1])
  my_map.drawparallels(np.arange(-90, 90, 30), labels =[1,0,0,0])
  plt.title("{:04d}-{:02d}-{:02d}-{:02d} Labels".format(year,month,day,time_step_index))
  cbar.ax.set_ylabel('TMQ kg $m^{-2}$')

  mask_ex = plt.gcf()
  #mask_ex.savefig("/global/cscratch1/sd/amahesh/segm_plots/combined_mask+{:04d}-{:02d}-{:02d}-{:02d}-{:02d}.png".format(year,month,day,time_step_index, run_num))
  mask_ex.savefig("{}{}combined_mask+{:04d}-{:02d}-{:02d}-{:02d}-{:02d}.png".format(cli_args.vis_output_dir, print_field, year,month,day,time_step_index, run_num))
  plt.clf()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--label_dir', type=str, help='directory of the label files')
  parser.add_argument('--year', type=str, help='the year of visualizations which you would like to out')
  parser.add_argument('--vis_output_dir', type=str, help='directory to output the label files')
  cli_args = parser.parse_args()

  with nc.Dataset("/global/cscratch1/sd/mwehner/machine_learning_climate_data/All-Hist/CAM5-1-0.25degree_All-Hist_est1_v3_run1/h2/CAM5-1-0.25degree_All-Hist_est1_v3_run1.cam.h2.2015-12-31-00000.nc") as fin:
      lats = fin['lat'][:]
      lons = fin['lon'][:]

  label_files = sorted(glob("{}*{}*".format(cli_args.label_dir, cli_args.year)))
  for i, label_file in enumerate(label_files):
    f = h5.File(label_file, 'r')
    save_data = f['climate']['data']
    save_label_0 = f['climate']['labels_0']
    save_label_1 = f['climate']['labels_1']

    if i % 50:
      print("{} of {} files done".format(i, len(label_files)))

    filename = os.path.basename(label_file)

    year = int(filename.split("-")[1])
    month = int(filename.split("-")[2])
    day = int(filename.split("-")[3])
    time_step_index = int(filename.split("-")[4])
    run_num = int(filename.split("-")[5][0])
    plot_mask(lons, lats, save_data[:,:,0], save_label_0,
           year, month, day, time_step_index, run_num, print_field="label_0")
    plot_mask(lons, lats, save_data[:,:,0], save_label_1,
           year, month, day, time_step_index, run_num, print_field="label_1")


