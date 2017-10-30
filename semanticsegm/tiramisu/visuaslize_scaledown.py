import matplotlib as mpl
mpl.use('agg')
import glob
import warnings
from skimage.transform import resize

import matplotlib.pyplot as plt
import numpy as np
import argparse
import netCDF4 as nc
from os import listdir
from os.path import isfile, join
import pandas as pd
import pickle

scale_down = 4
image_height = 768 / scale_down
image_width = 1152 / scale_down

def downsize_labels(lbl_array, image_height, image_width):
    temp_1 = np.copy(lbl_array)
    temp_2 = np.copy(lbl_array)

    import IPython; IPython.embed()

    temp_1[np.where(temp_1) != 1] = 0
    temp_2[np.where(temp_2) != 2] = 0
    temp_1 = resize(temp_1, (image_height, image_width))
    temp_2 = resize(temp_2, (image_height, image_width))
    new_array = np.zeros((image_height, image_width))
    new_array[np.where(temp_1) > 0] = 1
    new_array[np.where(temp_2) > 0] = 2
    return new_array

def plot_mask(lons, lats, img_array, storm_mask):
    # my_map = Basemap(projection='robin', llcrnrlat=min(lats), lon_0=180,
    #               llcrnrlon=min(lons), urcrnrlat=max(lats), urcrnrlon=max(lons), resolution = 'c')
 
    xx, yy = np.meshgrid(lons, lats)
    # x_map,y_map = my_map(xx,yy)
    # x_plot, y_plot = my_map(storm_lon, storm_lat)
    # my_map.drawcoastlines(color="black")
    plt.contourf(xx,yy,img_array,64,cmap='viridis')
    #my_map.plot(x_plot, y_plot, 'r*', color = "red")
    #cbar = my_map.colorbar()
    plt.contourf(xx,yy,storm_mask, alpha=0.42,cmap='gray')
    plt.title("TMQ with Segmented TECA Storms")
    #cbar.ax.set_ylabel('TMQ kg $m^{-2}$')

    mask_ex = plt.gcf()
    mask_ex.savefig("test5.png")
    plt.clf()


def load_lat_lon_TMQ(filepath, year, month, day, time_step):
    filepath += "CAM5-1-0.25degree_All-Hist_est1_v3_run2.cam.h2." +"{:04d}-{:02d}-{:02d}-00000.nc".format(year, month, day)
    print(filepath)
    with nc.Dataset(filepath) as fin:
        TMQ = fin['TMQ'][:][time_step]
        lons = fin['lon'][:]
        lats = fin['lat'][:]
        print(TMQ.shape)
    return lons[::scale_down], lats[::scale_down], (TMQ * 1000).astype('uint32')

def _process_img_id_string(img_id_string):
  year = int(img_id_string[:4])
  month = int(img_id_string[4:6])
  day = int(img_id_string[6:8])
  time_step = int(img_id_string[8:10])
  return year, month, day, time_step

warnings.filterwarnings("ignore",category=DeprecationWarning)

frames_path = '/home/mudigonda/Data/tiramisu_images/'

labels_path = '/home/mudigonda/Data/tiramisu_labels/[0-9]*'

fnames = glob.glob(frames_path+"*")
lnames = glob.glob(labels_path+"*")

imgs = np.stack([np.load(fn) for fn in fnames])
print(imgs.dtype)
imgs = np.asarray([resize(img, (image_height, image_width),preserve_range=True)for img in imgs]).astype('uint32')


labels = np.stack([np.load(fn) for fn in lnames]).astype('uint8')
#labels = np.asarray([downsize_labels(lbl, image_height, image_width)for lbl in labels])
labels = labels[:,::scale_down,::scale_down]
print((imgs.shape,labels.shape))


TMQ = imgs[len(imgs)-6]
label = labels[len(labels) - 6]
id_start_index = min([i for i, c in enumerate(fnames[len(fnames) - 5]) if c.isdigit()])
img_id = fnames[len(fnames) - 5][id_start_index:]
print(img_id)
year, month, day, time_step = _process_img_id_string(img_id)
lons, lats, _ = load_lat_lon_TMQ("/home/mudigonda/files_for_first_maskrcnn_test/", year, month, day, time_step)
import IPython; IPython.embed()
plot_mask(lons, lats, TMQ.squeeze(), label.squeeze())
