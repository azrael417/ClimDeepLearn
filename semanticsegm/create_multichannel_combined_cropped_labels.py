import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
import numpy as np
import argparse
from skimage.filters import threshold_otsu
import netCDF4 as nc
from os import listdir
from os.path import isfile, join
import pandas as pd
import geopy.distance
from scipy import ndimage
import glob
import os
import floodfillsearch.floodFillSearch as flood
from scipy.misc import imsave

from mpi4py import MPI


print("finished importing")

#UNCOMMENT THE FOLLOWING 2 LINES IF YOU WANT TO RUN IN PARALLEL
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

print("SIZE: " + str(size))
print("RANK: " + str(rank))

image_height = 768
image_width = 1152

#------------- AR DETECTION METHODS --------------#

#"blobs" are candidate ARs


def calculate_blob_length(blob):
  """Calculates the length of a blob"""
  min_lat = lats[min(blob[0])]
  min_lon = lons[min(blob[1])]
  max_lat = lats[max(blob[0])]
  max_lon = lons[max(blob[1])]
  return max(geopy.distance.great_circle((max_lat, max_lon),(min_lat, min_lon)).km, geopy.distance.great_circle((max_lat, max_lon),(min_lat, min_lon)).km)

#note: time_step must be a number from 0 to 7, inclusive.  CAM5 model output has 8 time steps.
def get_AR_blobs(U850, V850, QREFHT, time_step):
  #with nc.Dataset(filepath) as fin:

   #Calculate IVT
  IVT_u = U850 * QREFHT
  IVT_v = V850 * QREFHT
  IVT = np.sqrt(IVT_u**2 + IVT_v**2)

  #Calculate IVT anomalies
  IVT_time_percentile = 95
  IVT_threshold = np.percentile(IVT,IVT_time_percentile,axis=(1,2))[:,np.newaxis,np.newaxis]

  # damp anomalies to 0 near the tropics
  lon2d,lat2d = np.meshgrid(lons,lats)
  sigma_lat = 10 # degrees
  gaussian_band = 1 - np.exp(-lat2d**2/(2*sigma_lat**2))

  # calculate IVT anomalies
  IVT_anomaly = IVT*gaussian_band[np.newaxis,...] - IVT_threshold

  ivt_blobs = flood.floodFillSearch(IVT_anomaly[0])
  return ivt_blobs


def get_AR_semantic_mask(U850, V850, QREFHT, time_step, semantic_mask):
  ivt_blobs = get_AR_blobs(U850, V850, QREFHT, time_step)
  for blob in ivt_blobs:
      if calculate_blob_length(blob) > 1500:
        semantic_mask[blob] = 2
      
  #ivt_blob_random_array = np.ma.masked_less_equal(ivt_blob_random_array,0)
  return semantic_mask

#get the number of AR instances, and append lat/lon bounding box coordinatoes to "instance_boxes"
def get_num_AR_instances(U850, V850, QREFHT, time_step, instance_masks, instance_boxes, num_instances, lats, lons): 
  ivt_blobs = get_AR_blobs(U850, V850, QREFHT, time_step)
  for blob in ivt_blobs:
    if calculate_blob_length(blob) > 1500:
      ivt_blob_random_array = np.zeros((image_height,image_width))
      ivt_blob_random_array[blob] = 1
      instance_masks.append(ivt_blob_random_array)

      min_lat = min(blob[0])
      min_lon = min(blob[1])
      max_lat = max(blob[0])
      max_lon = max(blob[1])
      instance_boxes.append(np.asarray([min_lat, min_lon, max_lat, max_lon, 2]))
      num_instances+=1
  return num_instances

#------- Binarize tropical cyclone regions -------------#
def binarize(img_array, mask, lat_end, lat_start, lon_end, lon_start):
  im_slice = img_array[lat_start: lat_end, lon_start: lon_end]
  intersect = False
  if np.any(mask[lat_start: lat_end, lon_start: lon_end] > 0):
    intersect = True
  
  #Find the Otsu threshold of the image slice
  otsu_thresh = threshold_otsu(im_slice)
  #binary_adaptive = im_slice > otsu_thresh
  binary_adaptive = im_slice > np.percentile(im_slice, 80)
  #Find the largest contiguous region of areas above the otsu threshold
  def filter_isolated_cells(array, struct):
      """ Return array with completely isolated single cells removed
      :param array: Array with completely isolated single cells
        :param struct: Structure array for generating unique regions
      :return: Array with minimum region size < max_size
      """
      filtered_array = np.copy(array)
      id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
      id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids + 1)))
      area_mask = (id_sizes < np.amax(id_sizes))
      filtered_array[area_mask[id_regions]] = 0
      id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
      return filtered_array

  # Run function on sample array
  binary_adaptive = filter_isolated_cells(binary_adaptive, struct=np.ones((3,3)))

  # Plot output, with all isolated single cells removed
  #plt.imshow(filtered_array, cmap=plt.cm.gray, interpolation='nearest')

  #Mark the pixels in the mask as 1 where there are TCs
  temp = mask[lat_start: lat_end, lon_start: lon_end]
  temp[np.where(binary_adaptive > 0)] = 1
  mask[lat_start: lat_end, lon_start: lon_end] = temp
  #mask[lat_start: lat_end, lon_start: lon_end] = binary_adaptive.astype(int) 
  return mask, intersect

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def plot_mask(lons, lats, img_array, storm_mask, storm_lon, storm_lat,
        year, month, day, time_step_index):
  my_map = Basemap(projection='robin', llcrnrlat=min(lats), lon_0=np.median(lons),
                  llcrnrlon=min(lons), urcrnrlat=max(lats), urcrnrlon=max(lons), resolution = 'c')
 
  xx, yy = np.meshgrid(lons, lats)
  x_map,y_map = my_map(xx,yy)
  x_plot, y_plot = my_map(storm_lon, storm_lat)
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
  mask_ex.savefig("./plots_for_clipping/combined_mask+3{:04d}-{:02d}-{:02d}-{:02d}-{:02d}.png".format(year,month,day,time_step_index, progress_counter))
  plt.clf()

print("before loading teca_subtables")

#The TECA subtables are csv versions of the TECA output table (one subtable for each day that TECA was run) 
#The original TECA output table is in .bin format and in Karthik's scratch.
path_to_subtables = "/global/cscratch1/sd/amahesh/segmentation_labels/teca_subtables/*.csv"

teca_subtables = np.asarray([os.path.basename(x) for x in glob.glob(path_to_subtables)])
shuffle_indices = np.random.permutation(len(teca_subtables)) 
np.save("./shuffle_indices_glob_files.npy", shuffle_indices)
teca_subtables = teca_subtables[shuffle_indices]
print("loaded in teca_subtables")

#path_to_labels = "/global/cscratch1/sd/mayur/segm_labels/"
path_to_labels = "/global/cscratch1/sd/amahesh/segm_labels/"
path_to_CAM5_files = "/global/cscratch1/sd/mwehner/CAM5-1-0.25degree_All-Hist_est1_v3_run2/run/h2/CAM5-1-0.25degree_All-Hist_est1_v3_run2.cam.h2."

progress_counter = 0

for ii,table_name in enumerate(teca_subtables):
  print("I index: " + str(ii))
  #UNCOMMENT THE FOLLOWING LINE IF YOU WANT TO RUN IN PARALLEL
  if ii % size != rank: continue
  print(str(rank))

  year = int(table_name[12:16])
  month = int(table_name[17:19])
  day = int(table_name[20:22])

  id_string = "{:04d}{:02d}{:02d}".format(year,month,day)

  curr_table = pd.read_csv(path_to_subtables[:-5]+table_name)

  #Add 4 to the tropical cyclone radii (so that the radii aren't too small)
  #curr_table['r0'] = curr_table['r0'][:] + 4
  curr_table['r0'] = curr_table['r0'][:] + 4.25

  #time_step_index refers to the 8 snapshots of data available for each data.
  for time_step_index in range(8):
    #Read in the TMQ data for the corresponding year, month, day, and time_step 
    with nc.Dataset(path_to_CAM5_files+"{:04d}-{:02d}-{:02d}-00000.nc".format(year, month, day)) as fin:
      #Extract the fields of interest
      TMQ = fin['TMQ'][:][time_step_index]
      lats = fin['lat'][:]
      lons = fin['lon'][:]
      U850 = fin.variables['U850'][:][time_step_index]
      V850 = fin.variables['V850'][:][time_step_index]
      QREFHT = fin.variables['QREFHT'][:][time_step_index]
      PS = fin.variables['PS'][:][time_step_index]
      PSL = fin.variables['PSL'][:][time_step_index]
      T200 = fin.variables['T200'][:][time_step_index]
      T500 = fin.variables['T500'][:][time_step_index]
      PRECT = fin.variables['PRECT'][:][time_step_index]
      TS = fin.variables['TS'][:][time_step_index]
      TREFHT = fin.variables['TREFHT'][:][time_step_index]
      Z1000 = fin.variables['Z1000'][:][time_step_index]
      Z200 = fin.variables['Z200'][:][time_step_index]
      ZBOT = fin.variables['ZBOT'][:][time_step_index]
      UBOT = fin.variables['UBOT'][:][time_step_index]
      VBOT = fin.variables['VBOT'][:][time_step_index]
      #U850 = np.expand_dims(U850, axis=0)
      #V850 = np.expand_dims(V850, axis=0)
      #QREFHT = np.expand_dims(QREFHT, axis=0)

    #The semantic mask is one mask with 1's for TCs, 2's for ARs, and 0's everywhere else
    semantic_mask = np.zeros((image_height, image_width))
    print("loaded in variables")
    #For instance segmentation, there are N masks, where N = number of instances.  Each mask is an array of size height, width
    instance_masks = []
    #For instance segmentation, the ground truth boxes surrounding the storm are also stored.  They are in the format of (N, 5) --> (x1, y1, x2, y2, classid)
    instance_boxes = []
    #A list of booleans indicating corresponding semantic mask has an AR intersecting with a TC
    intersects =[]
    #Set the semantic mask to 0 where there are AR pixels
    semantic_mask_AR = get_AR_semantic_mask(np.expand_dims(U850,axis=0), np.expand_dims(V850,axis=0), np.expand_dims(QREFHT,axis=0),time_step_index, semantic_mask)
    num_instances = 0
    #Note: the following method will append one mask for each AR instance to "instance_masks."  This feature
    #will be useful for instance segmentation in the future. For ECCV, we are only doing semantic segmentation.
    num_instances = get_num_AR_instances(np.expand_dims(U850,axis=0), np.expand_dims(V850,axis=0), np.expand_dims(QREFHT,axis=0), time_step_index, instance_masks, instance_boxes, num_instances, lats, lons)
    #print("got AR masks")
    #time_step_index*3 yields 0,3,6,9,12,15,18,or21
    curr_table_time = curr_table[curr_table['hour'] == time_step_index*3]
    print("LENCURRTABLETIME: " + str(len(curr_table_time)))
    if len(curr_table_time) > 0:
      num_instances += len(curr_table_time)
      for index, row in curr_table_time.iterrows():
        #Find the lat/ lon start and end indices of the box around the tropical cyclone center
        lat_end_index = find_nearest(lats, row['lat'] + row['r0'])
        lat_start_index = find_nearest(lats, row['lat'] - row['r0'])
        lon_end_index = find_nearest(lons, row['lon'] + row['r0'])
        lon_start_index = find_nearest(lons, row['lon'] - row['r0'])

        if len(np.unique(TMQ[lat_start_index: lat_end_index, lon_start_index: lon_end_index])) > 1:
          #Set the relevant parts of the semantic_mask to 1, for TC
          semantic_mask_TMQ, intersect = binarize(TMQ, semantic_mask_AR.copy(), lat_end_index, lat_start_index, lon_end_index, lon_start_index)
          semantic_mask_U850, intersect = binarize(U850, semantic_mask_AR.copy(), lat_end_index, lat_start_index, lon_end_index, lon_start_index)
          semantic_mask_UBOT, intersect = binarize(UBOT, semantic_mask_AR.copy(), lat_end_index, lat_start_index, lon_end_index, lon_start_index)
          semantic_mask_V850, intersect = binarize(V850, semantic_mask_AR.copy(), lat_end_index, lat_start_index, lon_end_index, lon_start_index)
          semantic_mask_VBOT, intersect = binarize(VBOT, semantic_mask_AR.copy(), lat_end_index, lat_start_index, lon_end_index, lon_start_index)
          semantic_mask_QREFHT, intersect = binarize(QREFHT, semantic_mask_AR.copy(), lat_end_index, lat_start_index, lon_end_index, lon_start_index)
          semantic_mask_PS, intersect = binarize(PS, semantic_mask_AR.copy(), lat_end_index, lat_start_index, lon_end_index, lon_start_index)
          semantic_mask_PSL, intersect = binarize(PSL, semantic_mask_AR.copy(), lat_end_index, lat_start_index, lon_end_index, lon_start_index)
          semantic_mask_T200, intersect = binarize(T200, semantic_mask_AR.copy(), lat_end_index, lat_start_index, lon_end_index, lon_start_index)
          semantic_mask_T500, intersect = binarize(T500, semantic_mask_AR.copy(), lat_end_index, lat_start_index, lon_end_index, lon_start_index)
          semantic_mask_PRECT, intersect = binarize(PRECT, semantic_mask_AR.copy(), lat_end_index, lat_start_index, lon_end_index, lon_start_index)
          semantic_mask_TS, intersect = binarize(TS, semantic_mask_AR.copy(), lat_end_index, lat_start_index, lon_end_index, lon_start_index)
          semantic_mask_TREFHT, intersect = binarize(TREFHT, semantic_mask_AR.copy(), lat_end_index, lat_start_index, lon_end_index, lon_start_index)
          semantic_mask_Z1000, intersect = binarize(Z1000, semantic_mask_AR.copy(), lat_end_index, lat_start_index, lon_end_index, lon_start_index)
          semantic_mask_Z200, intersect = binarize(Z200, semantic_mask_AR.copy(), lat_end_index, lat_start_index, lon_end_index, lon_start_index)
          semantic_mask_ZBOT, intersect = binarize(ZBOT, semantic_mask_AR.copy(), lat_end_index, lat_start_index, lon_end_index, lon_start_index)
          #intersects.append(intersect)
        
      #The following if condition tests if the flood fill algorithm found any ARs
      if len(instance_masks) > 0:
        print('now saving')
        imsave(path_to_labels+"tmq_"+str(ii)+"_"+str(time_step_index)+".png",TMQ)
        imsave(path_to_labels+"mask_TMQ_"+str(ii)+"_"+str(time_step_index)+".png",semantic_mask_TMQ)
        imsave(path_to_labels+"mask_u850_"+str(ii)+"_"+str(time_step_index)+".png",semantic_mask_U850)
        imsave(path_to_labels+"mask_uBOT_"+str(ii)+"_"+str(time_step_index)+".png",semantic_mask_UBOT)
        imsave(path_to_labels+"mask_v850_"+str(ii)+"_"+str(time_step_index)+".png",semantic_mask_V850)
        imsave(path_to_labels+"mask_vBOT_"+str(ii)+"_"+str(time_step_index)+".png",semantic_mask_VBOT)
        imsave(path_to_labels+"mask_QREFHT_"+str(ii)+"_"+str(time_step_index)+".png",semantic_mask_QREFHT)
        imsave(path_to_labels+"mask_PS_"+str(ii)+"_"+str(time_step_index)+".png",semantic_mask_PS)
        imsave(path_to_labels+"mask_PSL_"+str(ii)+"_"+str(time_step_index)+".png",semantic_mask_PSL)
        imsave(path_to_labels+"mask_T200_"+str(ii)+"_"+str(time_step_index)+".png",semantic_mask_T200)
        imsave(path_to_labels+"mask_T500_"+str(ii)+"_"+str(time_step_index)+".png",semantic_mask_T500)
        imsave(path_to_labels+"mask_PRECT_"+str(ii)+"_"+str(time_step_index)+".png",semantic_mask_PRECT)
        imsave(path_to_labels+"mask_TS_"+str(ii)+"_"+str(time_step_index)+".png",semantic_mask_TS)
        imsave(path_to_labels+"mask_TREFHT_"+str(ii)+"_"+str(time_step_index)+".png",semantic_mask_TREFHT)
        imsave(path_to_labels+"mask_Z1000_"+str(ii)+"_"+str(time_step_index)+".png",semantic_mask_Z1000)
        imsave(path_to_labels+"mask_Z200_"+str(ii)+"_"+str(time_step_index)+".png",semantic_mask_Z200)
        imsave(path_to_labels+"mask_ZBOT_"+str(ii)+"_"+str(time_step_index)+".png",semantic_mask_ZBOT)
