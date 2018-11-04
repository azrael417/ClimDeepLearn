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
from copy import copy

from matplotlib.colors import ListedColormap
import sys
import os

def plot_mask_sean(img_array, storm_mask, fname, year, month, day):

  # Choose colormap
  cmap = mpl.cm.viridis

  # Get the colormap colors
  my_cmap = cmap(np.arange(cmap.N))

  # Set alpha
  alpha = np.linspace(0, 1, cmap.N)
  my_cmap[:,0] = (1-alpha) + alpha * my_cmap[:,0]
  my_cmap[:,1] = (1-alpha) + alpha * my_cmap[:,1]
  my_cmap[:,2] = (1-alpha) + alpha * my_cmap[:,2]

  # Create new colormap
  my_cmap = ListedColormap(my_cmap)
  
  # l = p['label'] / 100
  p = storm_mask #p['prediction']
  p = np.roll(p,[0,1152/2])
  p1 = (p == 100)
  p2 = (p == 200)

  d = img_array #h['climate']['data'][0,...]
  d = np.roll(d,[0,1152/2])

  lats = np.linspace(-90,90,768)
  longs = np.linspace(-180,180,1152)

  def do_fig(figsize, fname, year, month, day):
      fig = plt.figure(figsize=figsize)

      # my_map = Basemap(projection='eck4', lon_0=np.median(lons),
      #                  resolution = 'c')
      my_map = Basemap(projection='robin', llcrnrlat=min(lats), lon_0=np.median(longs),
                  llcrnrlon=min(longs), urcrnrlat=max(lats), urcrnrlon=max(longs), resolution = 'c')
      xx, yy = np.meshgrid(longs, lats)
      x_map,y_map = my_map(xx,yy)
      my_map.drawcoastlines(color=[0.5,0.5,0.5])
      my_map.contourf(x_map,y_map,d,64,cmap=my_cmap, vmax=89, vmin=0, levels=np.arange(0,89,2))
      cbar = my_map.colorbar(ticks=np.arange(0,89,11))
      cbar.ax.set_ylabel('Integrated Water Vapor kg $m^{-2}$')
      plt.title("Segmented Extreme Weather Patterns {}-{}-{}".format(year, month, day))
      if True:
          ar_contour = my_map.contour(x_map,y_map,p2,[0.5],linewidths=1,colors='blue', label='Atmospheric River')
          tc_contour = my_map.contour(x_map,y_map,p1,[0.5],linewidths=1,colors='red', label='Tropical Cyclone')
          

      lines = [tc_contour.collections[0], ar_contour.collections[0]]
      labels = ['Tropical Cyclone', "Atmospheric River"]
      my_map.drawmeridians(np.arange(-180, 180, 60), labels=[0,0,0,1])
      my_map.drawparallels(np.arange(-90, 90, 30), labels =[1,0,0,0])
      plt.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

      mask_ex = plt.gcf()
      mask_ex.savefig(fname,bbox_inches='tight')
      plt.clf()
  
  do_fig((10,7), fname, year, month, day)

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
  cmap = copy(plt.cm.get_cmap('bwr'))
  cmap.set_bad(alpha=0)
  # my_map.contourf(x_map,y_map,storm_mask, alpha=0.42,cmap='gray')
  my_map.drawmeridians(np.arange(-180, 180, 60), labels=[0,0,0,1])
  my_map.drawparallels(np.arange(-90, 90, 30), labels =[1,0,0,0])
  label_contour = my_map.contourf(x_map,y_map,np.ma.masked_less(storm_mask,0.009),cmap=cmap,vmin=1,vmax=2,alpha=0.3)
  # contour_cbar = my_map.colorbar(label_contour, location='bottom',pad="5%")
  # contour_cbar.ax.set_xlabel('Atmospheric River Confidence Index')
  plt.title("Integrated Water Vapor (IWV) with Segmented Extreme Weather {}-{}-{}".format(year, month, day))
  cbar.ax.set_ylabel('IWV kg $m^{-2}$')

  mask_ex = plt.gcf()
  #mask_ex.savefig("/global/cscratch1/sd/amahesh/segm_plots/combined_mask+{:04d}-{:02d}-{:02d}-{:02d}-{:02d}.png".format(year,month,day,time_step_index, run_num))
  #mask_ex.savefig("/global/cscratch1/sd/mayur/segm_plots/combined_mask+{:04d}-{:02d}-{:02d}-{:02d}-{:02d}.png".format(year,month,day,time_step_index, run_num))
  mask_ex.savefig(fname)
  plt.clf()


if __name__ == "__main__":
  AP = argparse.ArgumentParser()
  AP.add_argument("--datapath",type=str,default=None,required=True,help="Path in which to find the datafiles belonging to labels and predictions")
  AP.add_argument("--maskpath",type=str,default=None,required=True,help="Path in which to find the mask files from which to pull labels or predicted labels. Assumed to be in npz.")
  AP.add_argument("--outpath",type=str,default=None,required=True,help="Path in which to store the generated images")
  AP.add_argument("--lat_lon",type=str,default="lat_lon.pkl",help="Name of the pkl file from which to pull lats and lons. Assumed to be in pkl")
  parsed_args = AP.parse_args()
  with open(parsed_args.lat_lon,'rb') as f:
    lat_lon = pickle.load(f)
  print("loaded lat and lon")
   
  #get all the maskfiles
  maskfiles = [x for x in os.listdir(parsed_args.maskpath) if os.path.splitext(x)[1]==".npz"]
  
  if not maskfiles:
    raise ValueError("Error, there are no npz files in path {}.".format(parsed_args.maskpath))
  
  #create output path
  if not os.path.isdir(parsed_args.outpath):
    os.makedirs(parsed_args.outpath)
  
  #load mask file
  for maskfile in maskfiles:
    masks = np.load(os.path.join(parsed_args.maskpath, maskfile))
   
    #parse filenames
    fnames = [os.path.basename(x) for x in masks['filename']]
   
    #grab the data from the standard path
    for idx,fname in enumerate(fnames):
      with h5.File(os.path.join(parsed_args.datapath, fname), "r") as f:
        data = f["climate"]["data"][...]
        labels = f["climate"]["labels"][...]

      #do sanity checks
      if np.linalg.norm(masks["label"][idx,...]-labels*100) > 0.0001:
        print("Error, label inconsistencies detected between {npzf} and {h5f} files.".format(npzf=maskfile,h5f=fname))
        continue
     
      #do the plotting
      basename = os.path.join(parsed_args.outpath, os.path.splitext(fname)[0])
      TMQ = data[0]
      prediction = masks['prediction'][idx, ...]
      label = masks['label'][idx, ...]
      #plot TMQ prediction
      split_fname = basename.split("-")
      plot_mask_sean(TMQ, prediction, basename+"_tmq__prediction.png", 
        year=split_fname[1], month=split_fname[2], day=split_fname[3])
      #plot TMQ tkurth prediction (whatever that is)
      # plot_mask_sean(TMQ, label, basename+"_tmq_label_tkurth.png",
      #   year=split_fname[1], month=split_fname[2], day=split_fname[3])

