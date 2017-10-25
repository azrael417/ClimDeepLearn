import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
import numpy as np
import argparse
from skimage import segmentation
from skimage.future import graph
from skimage.filters import threshold_otsu, threshold_local, rank
from skimage.morphology import disk
import netCDF4 as nc
from os import listdir
from os.path import isfile, join
import pandas as pd
import pickle
import geopy.distance
from scipy import ndimage

import floodfillsearch.cFloodFillSearch as flood

def calculate_blob_length(blob):
		min_lat = lats[min(blob[0])]
		min_lon = lons[min(blob[1])]
		max_lat = lats[max(blob[0])]
		max_lon = lons[max(blob[1])]

		return geopy.distance.great_circle((max_lat, max_lon),(min_lat, min_lon)).km

#note: time_step must be a number from 0 to 7, inclusive
def get_AR_blobs(filepath, time_step):
	with nc.Dataset(filepath) as fin:
	    
	    U850 = fin.variables['U850'][:][time_step:time_step+1, :, :]
	    V850 = fin.variables['V850'][:][time_step:time_step+1, :, :]
	    QREFHT = fin.variables['QREFHT'][:][time_step:time_step+1, :, :]
	    lats = fin.variables['lat'][:]
	    lons = fin.variables['lon'][:]

	 #Calculate IVT
	IVT_u = U850 * QREFHT
	IVT_v = V850 * QREFHT
	IVT = np.sqrt(IVT_u**2 + IVT_v**2)

	#Calculate IVT anomalies
	# calculate the X-percentile anomalies for each timestep
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


def get_AR_semantic_mask(filepath, time_step, semantic_mask):
	ivt_blobs = get_AR_blobs(filepath, time_step)
	for blob in ivt_blobs:
	    if calculate_blob_length(blob) > 1500:
	    	semantic_mask[blob] = 2
	    
	#ivt_blob_random_array = np.ma.masked_less_equal(ivt_blob_random_array,0)
	return semantic_mask

def get_AR_instance_masks(filepath, time_step, instance_masks, instance_boxes, num_instances, lats, lons):
	
	ivt_blobs = get_AR_blobs(filepath, time_step)
	for blob in ivt_blobs:
		if calculate_blob_length(blob) > 1500:
			ivt_blob_random_array = np.zeros((768,1152))
			ivt_blob_random_array[blob] = 1
			instance_masks.append(ivt_blob_random_array)

			min_lat = lats[min(blob[0])]
			min_lon = lons[min(blob[1])]
			max_lat = lats[max(blob[0])]
			max_lon = lons[max(blob[1])]
			instance_boxes.append(np.asarray([min_lat, min_lon, max_lat, max_lon, 2]))
			num_instances+=1
	return num_instances

#unsupervised technique - binarize a portion of the image by setting the mask to either 1 or 0
def binarize(img_array, mask, lat_end, lat_start, lon_end, lon_start):
	im_slice = img_array[lat_start: lat_end, lon_start: lon_end]
	#adaptive_thresh = threshold_local(im_slice,block_size,offset=25)
    #adaptive_thresh = rank.otsu((im_slice - im_slice.mean())/im_slice.max(),disk(5))
	adaptive_thresh = threshold_otsu(im_slice)
	binary_adaptive = im_slice > adaptive_thresh

	#Select the largest contiguous region
	def filter_isolated_cells(array, struct):
	    """ Return array with completely isolated single cells removed
	    :param array: Array with completely isolated single cells
	    :param struct: Structure array for generating unique regions
	    :return: Array with minimum region size < max_size
	    """
	    filtered_array = np.copy(array)
	    id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
	    print(num_ids)
	    id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids + 1)))
	    area_mask = (id_sizes < np.amax(id_sizes))
	    filtered_array[area_mask[id_regions]] = 0
	    return filtered_array

	# Run function on sample array
	binary_adaptive = filter_isolated_cells(binary_adaptive, struct=np.ones((3,3)))

	# Plot output, with all isolated single cells removed
	#plt.imshow(filtered_array, cmap=plt.cm.gray, interpolation='nearest')

	mask[lat_start: lat_end, lon_start: lon_end] = binary_adaptive
	return mask

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def plot_mask(lons, lats, img_array, storm_mask, storm_lon, storm_lat,
			  year, month, day, time_step_index):
	my_map = Basemap(projection='robin', llcrnrlat=min(lats), lon_0=180,
                  llcrnrlon=min(lons), urcrnrlat=max(lats), urcrnrlon=max(lons), resolution = 'c')
 
	xx, yy = np.meshgrid(lons, lats)
	x_map,y_map = my_map(xx,yy)
	x_plot, y_plot = my_map(storm_lon, storm_lat)
	my_map.drawcoastlines(color="black")
	my_map.contourf(x_map,y_map,img_array,64,cmap='viridis')
	#my_map.plot(x_plot, y_plot, 'r*', color = "red")
	cbar = my_map.colorbar()
	my_map.contourf(x_map,y_map,storm_mask, alpha=0.42,cmap='gray')
	plt.title("TMQ with Segmented TECA Storms")
	cbar.ax.set_ylabel('TMQ kg $m^{-2}$')

	mask_ex = plt.gcf()
	mask_ex.savefig("./sample_seg_masks/combined_mask+3{:04d}-{:02d}-{:02d}-{:02d}.png".format(year,month,day,time_step_index))
	plt.clf()

#The TECA subtables are the chopped up version of the .bin file (1 TECA subtable for each day).  The subtables are in .csv form
path_to_subtables = "/global/cscratch1/sd/amahesh/segmentation_labels/teca_subtables/"
teca_subtables = [f for f in listdir(path_to_subtables) if isfile(join(path_to_subtables, f))]


path_to_labels = "/global/cscratch1/sd/amahesh/segmentation_labels/"
path_to_CAM5_files = "/global/cscratch1/sd/mwehner/CAM5-1-0.25degree_All-Hist_est1_v3_run2/run/h2/CAM5-1-0.25degree_All-Hist_est1_v3_run2.cam.h2."


for table_name in teca_subtables:
	year = int(table_name[12:16])
	month = int(table_name[17:19])
	day = int(table_name[20:22])
	curr_table = pd.read_csv(path_to_subtables+table_name)

	#Add 3 to the storm radii
	curr_table['r0'] = curr_table['r0'][:] + 3

	#time_step_index refers to the 8 snapshots of data available for each data.
	for time_step_index in range(8):
		#Read in the TMQ data for the corresponding year, month, day, and time_step 
		with nc.Dataset(path_to_CAM5_files+"{:04d}-{:02d}-{:02d}-00000.nc".format(year, month, day)) as fin:
			TMQ = fin['TMQ'][:][time_step_index]
			lats = fin['lat'][:]
			lons = fin['lon'][:]
		
		#The semantic mask is one mask with 1's where the storms are and 0's everywhere else
		semantic_mask = np.zeros((768,1152))
		
		#For instance segmentation, there are N masks, where N = number of instances.  Each mask is an array of size height, width
		instance_masks = []

		#For instance segmentation, the ground truth boxes surrounding the storm are also stored.  They are in the format of (N, 5) --> (x1, y1, x2, y2, classid)
		instance_boxes = []
		

		semantic_mask = get_AR_semantic_mask(path_to_CAM5_files+"{:04d}-{:02d}-{:02d}-00000.nc".format(year, month, day),time_step_index, semantic_mask)

		num_instances = 0
		num_instances = get_AR_instance_masks(path_to_CAM5_files+"{:04d}-{:02d}-{:02d}-00000.nc".format(year, month, day), time_step_index, instance_masks, instance_boxes, num_instances, lats, lons)

		#time_step_index*3 yields 0,3,6,9,12,15,18,or21
		curr_table_time = curr_table[curr_table['hour'] == time_step_index*3]
		num_instances = len(curr_table_time)
		if len(curr_table_time > 0):
			for index, row in curr_table_time.iterrows():
				#Boolean determining whether or not to calculate the threshold of this storm.  
				#If the storm is too small (and doesn't span multiple pixels), don't calculate its threshold
				lat_end_index = find_nearest(lats, row['lat'] + row['r0'])
				lat_start_index = find_nearest(lats, row['lat'] - row['r0'])
				lon_end_index = find_nearest(lons, row['lon'] + row['r0'])
				lon_start_index = find_nearest(lons, row['lon'] - row['r0'])

				if len(np.unique(TMQ[lat_start_index: lat_end_index, lon_start_index: lon_end_index])) > 1:
					#Set the relevant parts of the semantic_mask to 1
					semantic_mask = binarize(TMQ, semantic_mask, lat_end_index, lat_start_index, lon_end_index, lon_start_index)
					
					#Create a new mask for each instance, and set the relevant parts to 1
					instance_mask = np.zeros((768,1152))
					instance_mask = binarize(TMQ, instance_mask, lat_end_index, lat_start_index, lon_end_index, lon_start_index)
					instance_masks.append(instance_mask)

					#The ground trouth boxes are in the form x1, y1, x2, y2, class_id
					lat_end = row['lat'] + row['r0']
					lat_start = row['lat'] - row['r0']
					lon_end = row['lon'] + row['r0']
					lon_start = row['lon'] - row['lon']
					instance_boxes.append(np.asarray([lat_start, lon_start, lat_end, lon_end, 1]))

			

		
		if len(instance_masks) > 0:

				#Plot sample semantic mask
				plot_mask(lons, lats, TMQ, semantic_mask, row['lon'], row['lat'], year, month, day, time_step_index)

				np.save(path_to_labels+"semantic_combined_labels/{:04d}-{:02d}-{:02d}-{:02d}.npy".format(year,month,day,time_step_index), semantic_mask)
				instance_labels = [num_instances,np.asarray(instance_boxes), np.asarray(instance_masks)]
				with open(path_to_labels+"instance_combined_labels/{:04d}-{:02d}-{:02d}-{:02d}.pkl".format(year,month,day,time_step_index),'w') as f:
					pickle.dump(instance_labels,f)

