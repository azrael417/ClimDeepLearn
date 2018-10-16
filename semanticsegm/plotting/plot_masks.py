import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import h5py as h5
from mpl_toolkits.basemap import Basemap
import numpy as np
import argparse
from skimage.filters import threshold_otsu
#import geopy.distance
from scipy import ndimage
import glob
import os
from scipy.misc import imsave
import h5py as h5
import pickle


def plot_mask(lons, lats, img_array, storm_mask,fname):
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
  plt.title("TMQ with Segmented TECA Storms")
  cbar.ax.set_ylabel('TMQ kg $m^{-2}$')

  mask_ex = plt.gcf()
  #mask_ex.savefig("/global/cscratch1/sd/amahesh/segm_plots/combined_mask+{:04d}-{:02d}-{:02d}-{:02d}-{:02d}.png".format(year,month,day,time_step_index, run_num))
  #mask_ex.savefig("/global/cscratch1/sd/mayur/segm_plots/combined_mask+{:04d}-{:02d}-{:02d}-{:02d}-{:02d}.png".format(year,month,day,time_step_index, run_num))
  mask_ex.savefig(fname)
  plt.clf()


if __name__ == "__main__":
   AP = argparse.ArgumentParser()
   AP.add_argument("--datapath",type=str,default=None,required=True,help="Name path in which to find the datafiles belonging to labels and predictions")
   AP.add_argument("--masks",type=str,default=None,required=True,help="Name of the masks file from which to pull labels or predicted labels. Assumed to be in npz")
   AP.add_argument("--lat_lon",type=str,default="lat_lon.pkl",help="Name of the pkl file from which to pull lats and lons. Assumed to be in pkl")
   parsed_args = AP.parse_args()
   with open(parsed_args.lat_lon,'rb') as f:
     lat_lon = pickle.load(f, encoding='latin1')
   print("loaded lat and lon")
   print(parsed_args.masks)
   
   #load mask file
   masks = np.load(parsed_args.masks)
   
   #parse filenames
   fnames = [os.path.basename(x) for x in masks['filename']]
   
   #grab the data from the standard path
   for idx,fname in enumerate(fnames):
     print(fname)
     with h5.File(os.path.join(parsed_args.datapath, fname), "r") as f:
       data = f["climate"]["data"][...]
       labels = f["climate"]["labels"][...]
       
       print(labels.shape)
       
     #do sanity checks
     print(np.linalg.norm(masks["label"][idx,...]-labels*100))
     
     #import IPython; IPython.embed()     
   
     #TMQ = data['climate']['data'][0]
   #plot_mask(lat_lon['lon'],lat_lon['lat'],TMQ,masks['prediction'],fname+"_tmq__prediction.png")
   #plot_mask(lat_lon['lon'],lat_lon['lat'],TMQ,masks['label'],fname+"_tmq_label_tkurth.png")
   #plot_mask(lat_lon['lon'],lat_lon['lat'],TMQ,labels['climate']['labels'],fname+"_tmq_label_amahesh.png")
   #plot_mask(lat_lon['lon'],lat_lon['lat'],TMQ,data['climate']['labels'],fname+"_tmq_label_data.png")
