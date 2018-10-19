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
from tqdm import tqdm

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
  AP.add_argument("--datapath",type=str,default=None,required=True,help="Path in which to find the datafiles belonging to labels and predictions")
  AP.add_argument("--maskpath",type=str,default=None,required=True,help="Path in which to find the mask files from which to pull labels or predicted labels. Assumed to be in npz.")
  AP.add_argument("--outpath",type=str,default=None,required=True,help="Path in which to store the generated images")
  AP.add_argument("--lat_lon",type=str,default="lat_lon.pkl",help="Name of the pkl file from which to pull lats and lons. Assumed to be in pkl")
  parsed_args = AP.parse_args()
  with open(parsed_args.lat_lon,'rb') as f:
    lat_lon = pickle.load(f, encoding='latin1')
  print("loaded lat and lon")
   
  #get all the maskfiles
  maskfiles = [x for x in os.listdir(parsed_args.maskpath) if os.path.splitext(x)[1]==".npz"]
  
  if not maskfiles:
    raise ValueError("Error, there are no npz files in path {}.".format(parsed_args.maskpath))
  
  #create output path
  if not os.path.isdir(parsed_args.outpath):
    os.makedirs(parsed_args.outpath)
  
  #load mask file
  for maskfile in tqdm(maskfiles):
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
      plot_mask(lat_lon['lon'], lat_lon['lat'], TMQ, prediction, basename+"_tmq__prediction.png")
      #plot TMQ tkurth prediction (whatever that is)
      plot_mask(lat_lon['lon'], lat_lon['lat'],TMQ, label, basename+"_tmq_label_tkurth.png")

