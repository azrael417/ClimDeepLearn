import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
import numpy as np
import argparse
# from skimage.filters import threshold_otsu
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
import h5py
import argparse

image_height = 768
image_width = 1152

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--label_output_dir', type=str, help='directory to output the label files')
  parser.add_argument('--dataset', type=str, help='select from the following: HAPPI20, HAPPI15, All-Hist')
  parser.add_argument('--vis_output_dir', type=str, help='directory to output visualizations of the label files')
  parser.add_argument('--parallel', action='store_true', help='whether or not to create the dataset in parallel')
  cli_args = parser.parse_args()

if cli_args.parallel:
  from mpi4py import MPI
  try:
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
  except:
    rank = 1
    size = 1

  print("SIZE: " + str(size))
  print("RANK: " + str(rank))

#------------- AR DETECTION METHODS --------------#

#"blobs" are candidate ARs

def calculate_area_grid_cells():
  """
  Calculate the area of each grid cell in the lat/ lon grid
  """
  dlat = np.diff(np.radians(lats))
  dlon = np.diff(np.radians(lons))
  dlat = np.append(dlat, dlat[0])
  dlon = np.append(dlon, dlon[0])
  dlon, dlat = np.meshgrid(dlon, dlat)
  return np.abs(6170**2 * (np.cos(np.radians(lats)) * (dlat * dlon).T).T)



def calculate_blob_length_and_width(blob):
  """Returns the length and width of the blob as a tuple"""
  min_lat = lats[min(blob[0])]
  min_lon = lons[min(blob[1])]
  max_lat = lats[max(blob[0])]
  max_lon = lons[max(blob[1])]


  length = max(geopy.distance.great_circle((max_lat, max_lon),(min_lat, max_lon)).km, geopy.distance.great_circle((max_lat, min_lon),(min_lat, min_lon)).km) * 1.0
  width = max(geopy.distance.great_circle((max_lat, max_lon),(max_lat, min_lon)).km, geopy.distance.great_circle((min_lat, max_lon),(min_lat, min_lon)).km) * 1.0
  diagonal = geopy.distance.great_circle((max_lat, max_lon),(min_lat, min_lon)).km
  return length, width, diagonal
  # return max(geopy.distance.great_circle((max_lat, max_lon),(min_lat, min_lon)).km, geopy.distance.great_circle((max_lat, max_lon),(min_lat, min_lon)).km)

#note: time_step must be a number from 0 to 7, inclusive.  CAM5 model output has 8 time steps.
def get_AR_blobs_TMQ(TMQ, time_step, ar_threshold_value):
  #with nc.Dataset(filepath) as fin:
  # TMQ_time_percentile = 72
  # TMQ_threshold = np.percentile(TMQ,TMQ_time_percentile,axis=(1,2))[:,np.newaxis,np.newaxis]
  TMQ_threshold = ar_threshold_value
  # from scipy import stats
  # print(stats.percentileofscore(TMQ.flatten(), TMQ_threshold))

  # damp anomalies to 0 near the tropics
  lon2d,lat2d = np.meshgrid(lons,lats)
  sigma_lat = 19 # degrees
  gaussian_band = 1 - np.exp(-lat2d**2/(2*sigma_lat**2))

  # calculate IVT anomalies
  TMQ_anomaly = TMQ*gaussian_band[np.newaxis,...] - TMQ_threshold

  ivt_blobs = flood.floodFillSearch(TMQ_anomaly[0])
  return ivt_blobs


def get_AR_semantic_mask(U850, V850, QREFHT, TMQ, time_step, semantic_mask, ar_threshold_value=None):
  ivt_blobs = get_AR_blobs_TMQ(TMQ, time_step, ar_threshold_value)
  ar_detected_bool = False

  # print("Candidate ARs: {}".format(len(ivt_blobs)))
  num_chosen_ARs = 0
  num_diagonal_thresh_ARs = 0
  num_length_thresh_ARs = 0

  for blob in ivt_blobs:
      #if calculate_blob_length(blob) > 1500:
      length, width, diagonal = calculate_blob_length_and_width(blob)
      if length > 0 and width > 0:

        #an approximation of the length of the ar is the diagonal of the bounding box
        length_bool = diagonal > 1500
        if length_bool: num_length_thresh_ARs+=1

        width_bool = np.sum(calculate_area_grid_cells()[blob]) / length < 1000
        # ratio_bool = length / width > 1.2 or length / width < 0.5
        if length_bool and width_bool:
          semantic_mask[blob] = 2
          ar_detected_bool = True
          num_chosen_ARs+= 1
  # print("Candidate ARs after diagonal threshold: {}".format(num_diagonal_thresh_ARs))
  # print("Candidate ARs after diagonal and length threshold: {}".format(num_length_thresh_ARs))
  # print("Selected ARs after length, width, and diagonal threshold: {}".format(num_chosen_ARs))
  #ivt_blob_random_array = np.ma.masked_less_equal(ivt_blob_random_array,0)
  return semantic_mask, ar_detected_bool



#get the number of AR instances, and append lat/lon bounding box coordinatoes to "instance_boxes"
def get_num_AR_instances(U850, V850, QREFHT, time_step, instance_masks, instance_boxes, num_instances, lats, lons): 
  ivt_blobs = get_AR_blobs(U850, V850, QREFHT, time_step)
  for blob in ivt_blobs:
    if calculate_blob_length_and_width(blob) > 1500:
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

#------- Binarize tropical cyclone regions -------------#
def binarize(img_array, mask, lat_end, lat_start, lon_end, lon_start):
  im_slice = img_array[lat_start: lat_end, lon_start: lon_end]
  intersect = False
  if np.any(mask[lat_start: lat_end, lon_start: lon_end] > 0):
    intersect = True
  
  #Find the Otsu threshold of the image slice
  # otsu_thresh = threshold_otsu(im_slice)
  #binary_adaptive = im_slice > otsu_thresh
  binary_adaptive = im_slice > np.percentile(im_slice, 93)
  #binary_adaptive = im_slice > np.percentile(im_slice, 89)
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


def extract_label_mask(year, month, time_step_index, run_num, U850, V850, QREFHT, TMQ, path_to_subtables, table_name, ar_threshold_value):
  semantic_mask = np.zeros((image_height, image_width))

  semantic_mask_AR, ar_detected_bool = get_AR_semantic_mask(np.expand_dims(U850,axis=0), np.expand_dims(V850,axis=0), np.expand_dims(QREFHT,axis=0), 
    np.expand_dims(TMQ, axis=0), time_step_index, semantic_mask, ar_threshold_value)
  

  curr_table = pd.read_csv("{}/{}".format(os.path.dirname(path_to_subtables), table_name))

  #Add 6.6 to the tropical cyclone radii (so that the radii aren't too small)
  #curr_table['r0'] = curr_table['r0'][:] + 4
  curr_table['r0'] = curr_table['r0'][:] + 6.6

  #time_step_index*3 yields 0,3,6,9,12,15,18,or21
  curr_table_time = curr_table[curr_table['hour'] == time_step_index*3]
  semantic_mask_combined = semantic_mask_AR.copy()
  if len(curr_table_time) > 0:
    #initialize the TC binarization masks
    semantic_mask_TMQ = semantic_mask_AR.copy()
    semantic_mask_U850 = semantic_mask_AR.copy()
    semantic_mask_V850 = semantic_mask_AR.copy()
    semantic_mask_PRECT = semantic_mask_AR.copy()

    for index, row in curr_table_time.iterrows():
      #Find the lat/ lon start and end indices of the box around the tropical cyclone center
      lat_end_index = find_nearest(lats, row['lat'] + row['r0'])
      lat_start_index = find_nearest(lats, row['lat'] - row['r0'])
      lon_end_index = find_nearest(lons, row['lon'] + row['r0'])
      lon_start_index = find_nearest(lons, row['lon'] - row['r0'])

      if len(np.unique(TMQ[lat_start_index: lat_end_index, lon_start_index: lon_end_index])) > 1:
        #Set the relevant parts of the semantic_mask to 1, for TC
        semantic_mask_TMQ, intersect = binarize(TMQ, semantic_mask_TMQ, lat_end_index, lat_start_index, lon_end_index, lon_start_index)
        semantic_mask_U850, intersect = binarize(U850, semantic_mask_U850, lat_end_index, lat_start_index, lon_end_index, lon_start_index)
        semantic_mask_V850, intersect = binarize(V850, semantic_mask_V850, lat_end_index, lat_start_index, lon_end_index, lon_start_index)
        semantic_mask_PRECT, intersect = binarize(PRECT, semantic_mask_PRECT, lat_end_index, lat_start_index, lon_end_index, lon_start_index)

        #combining channels 
        tmq_idx = np.where(semantic_mask_TMQ==1)
        u850_idx = np.where(semantic_mask_U850==1)
        prect_idx = np.where(semantic_mask_PRECT==1)
        v850_idx = np.where(semantic_mask_V850==1)


        #Make labels such that there are no disjoint regions and AR labels in the TC bounding box are not set to 0
        concat_idx = np.array([np.concatenate((tmq_idx[0],u850_idx[0],prect_idx[0],v850_idx[0])),np.concatenate((tmq_idx[1],u850_idx[1],prect_idx[1],v850_idx[1]))])
        semantic_mask_combined = semantic_mask_TMQ.copy()
        #semantic_mask_combined[concat_idx[0],concat_idx[1]] = 1.

        temp = np.zeros((768,1152))
        temp[concat_idx[0],concat_idx[1]] = 1
        temp = filter_isolated_cells(temp[lat_start_index: lat_end_index, lon_start_index: lon_end_index], struct=np.ones((3,3)))

        semantic_mask_combined[lat_start_index: lat_end_index, lon_start_index: lon_end_index][np.where(temp == 1)] = 1

  save_labels = semantic_mask_combined
  background_class = np.sum(semantic_mask==0)/(768*1152.0)
  tc_class = np.sum(semantic_mask_combined==1)/(768*1152.0)
  ar_class = np.sum(semantic_mask_combined==2)/(768*1152.0)
  save_labels_stats = np.asarray([np.mean(semantic_mask_combined), np.max(semantic_mask_combined), np.min(semantic_mask_combined),np.std(semantic_mask_combined),background_class, tc_class, ar_class])
  save_labels_stats = save_labels_stats.reshape((7,1))

  return save_labels, save_labels_stats, background_class < 0.9


progress_counter = 0

#print("before loading teca_subtables")

#The TECA subtables are csv versions of the TECA output table (one subtable for each day that TECA was run) 
#The original TECA output table is in .bin format and in Karthik's scratch.

path_to_sample_subtables = '/global/cscratch1/sd/amahesh/gb_helper/{}/label_0/subtables/*.csv'.format(cli_args.dataset)

sample_teca_subtables = np.sort(np.asarray([os.path.basename(x) for x in glob.glob(path_to_sample_subtables)]))
# np.random.seed(0)
# shuffle_indices = np.random.permutation(len(teca_subtables)) 
# teca_subtables = teca_subtables[shuffle_indices]
#print("loaded in teca_subtables")

for ii,table_name in enumerate(sample_teca_subtables):
  if cli_args.parallel:
    if ii % size != rank: continue

  year = int(table_name[12:16])
  month = int(table_name[17:19])
  day = int(table_name[20:22])
  run_num = int(table_name[-5:-4])

  # if ii % 30 == 0:
    # print("Current year: {}; Current month: {}; Current run number: {}".format(year, month, run_num))
  
  if cli_args.dataset == 'HAPPI15':
    path_to_CAM5_files = "/global/cscratch1/sd/mwehner/machine_learning_climate_data/HAPPI15/fvCAM5_HAPPI15_run" +str(run_num) + "/h2/fvCAM5_HAPPI15_run" + str(run_num) + ".cam.h2."
  elif cli_args.dataset == 'HAPPI20':
    path_to_CAM5_files = "/global/cscratch1/sd/mwehner/machine_learning_climate_data/HAPPI20/fvCAM5_HAPPI20_run" +str(run_num) + "/h2/fvCAM5_HAPPI20_run" + str(run_num) + ".cam.h2."
  elif cli_args.dataset == 'All-Hist':
    path_to_CAM5_files = "/global/cscratch1/sd/mwehner/machine_learning_climate_data/All-Hist/CAM5-1-0.25degree_All-Hist_est1_v3_run" + str(run_num) + "/h2/" + "CAM5-1-0.25degree_All-Hist_est1_v3_run" + str(run_num) + ".cam.h2."  

  #time_step_index refers to the 8 snapshots of data available for each data.
  for time_step_index in range(8):
    #Read in the TMQ data for the corresponding year, month, day, and time_step 
    try:
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

      # print("read in variables")
    except:
      print("Could not load in {} Dataset Year: {} Month: {} Day: {} Run_Num: {}".format(cli_args.dataset, year, month, day, run_num))
      continue
    #The semantic mask is one mask with 1's for TCs, 2's for ARs, and 0's everywhere else


    channel_list = [TMQ, U850, V850, UBOT, VBOT, QREFHT, PS, PSL, T200, T500, PRECT, TS, TREFHT, Z1000, Z200, ZBOT]

    save_data = np.stack(channel_list, axis=-1)
    #print(save_data.shape)
    save_data_stats = np.zeros((4,16))
    for channel_index, channel in enumerate(channel_list):
      save_data_stats[:,channel_index] = np.asarray([np.mean(channel), np.max(channel), np.min(channel),np.std(channel)])
    
    save_label_0, save_label_0_stats, label_0_anomaly_bool = extract_label_mask(year, month, time_step_index, run_num, U850, V850, QREFHT, TMQ, "/global/cscratch1/sd/amahesh/gb_helper/{}/label_0/subtables/".format(cli_args.dataset), table_name, 24)
    save_label_1, save_label_1_stats, label_1_anomaly_bool = extract_label_mask(year, month, time_step_index, run_num, U850, V850, QREFHT, TMQ, "/global/cscratch1/sd/amahesh/gb_helper/{}/label_1/subtables/".format(cli_args.dataset), table_name, 21)
    # save_label_2, save_label_2_stats = extract_label_mask(year, month, time_step_index, run_num, U850, V850, QREFHT, TMQ, path_to_subtables, table_name, 18)

    if label_0_anomaly_bool or label_1_anomaly_bool:
      anomaly_str = "-ANOMALY"
    else:
      anomaly_str = ""

    #try:
    #f = h5py.File("/global/cscratch1/sd/amahesh/segm_h5_v3_HAPPI15/data-{:04d}-{:02d}-{:02d}-{:02d}-{:01d}.h5".format(year,month,day, time_step_index, run_num),"w")
    f = h5py.File("{}data-{:04d}-{:02d}-{:02d}-{:02d}-{:01d}{}.h5".format(cli_args.label_output_dir, year,month,day, time_step_index, run_num, anomaly_str),"w")
    grp = f.create_group("climate")
    grp.create_dataset("data",(768,1152,16),dtype="f",data=save_data)
    grp.create_dataset("data_stats",(4,16),dtype="f",data=save_data_stats)
    #f.close()

    grp.create_dataset("labels_0",(768,1152),dtype="f",data=save_label_0)
    grp.create_dataset("labels_0_stats",(7,1),dtype="f",data=save_label_0_stats)
    grp.create_dataset("labels_1",(768,1152),dtype="f",data=save_label_1)
    grp.create_dataset("labels_1_stats",(7,1),dtype="f",data=save_label_1_stats)
    # grp.create_dataset("labels_2",(768,1152),dtype="f",data=save_label_2)
    # grp.create_dataset("labels_2_stats",(7,1),dtype="f",data=save_label_2_stats)
    f.close()  
    
    print("saved file: Year: {}: Month: {} Day: {} Time_Step_Index: {} run_num {}".format(year, month, day, time_step_index, run_num))
    if ii % 100 == 0 or year==2106 or year ==2110 or year==2115:
      plot_mask(lons, lats, save_data[:,:,0], save_label_0,
           year, month, day, time_step_index, run_num, print_field="label_0")
      plot_mask(lons, lats, save_data[:,:,0], save_label_1,
           year, month, day, time_step_index, run_num, print_field="label_1")
print("finished")
