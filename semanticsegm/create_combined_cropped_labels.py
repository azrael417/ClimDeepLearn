import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
import numpy as np
import argparse
#from skimage import segmentation
#from skimage.future import graph
from skimage.filters import threshold_otsu
import netCDF4 as nc
from os import listdir
from os.path import isfile, join
import pandas as pd
import geopy.distance
from scipy import ndimage
import glob

import floodfillsearch.cFloodFillSearch as flood

from mpi4py import MPI

print("finished importing")

#UNCOMMENT THE FOLLOWING 2 LINES IF YOU WANT TO RUN IN PARALLEL
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

scale_down = 1
image_height = 768 / scale_down
image_width = 1152 / scale_down

def calculate_blob_length(blob):
		min_lat = lats[min(blob[0])]
		min_lon = lons[min(blob[1])]
		max_lat = lats[max(blob[0])]
		max_lon = lons[max(blob[1])]

		return max(geopy.distance.great_circle((max_lat, max_lon),(min_lat, min_lon)).km, geopy.distance.great_circle((max_lat, max_lon),(min_lat, min_lon)).km)

#note: time_step must be a number from 0 to 7, inclusive
def get_AR_blobs(U850, V850, QREFHT, time_step):
	#with nc.Dataset(filepath) as fin:
	    
	    

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


def get_AR_semantic_mask(U850, V850, QREFHT, time_step, semantic_mask):
	ivt_blobs = get_AR_blobs(U850, V850, QREFHT, time_step)
	for blob in ivt_blobs:
	    if calculate_blob_length(blob) > 1500:
	    	semantic_mask[blob] = 2
	    
	#ivt_blob_random_array = np.ma.masked_less_equal(ivt_blob_random_array,0)
	return semantic_mask

def get_AR_instance_masks(U850, V850, QREFHT, time_step, instance_masks, instance_boxes, num_instances, lats, lons):
	
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

#unsupervised technique - binarize a portion of the image by setting the mask to either 1 or 0
def binarize(img_array, mask, lat_end, lat_start, lon_end, lon_start):
	im_slice = img_array[lat_start: lat_end, lon_start: lon_end]
	intersect = False
	if np.any(mask[lat_start: lat_end, lon_start: lon_end] > 0):
		intersect = True
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
	    id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids + 1)))
	    area_mask = (id_sizes < np.amax(id_sizes))
	    filtered_array[area_mask[id_regions]] = 0
	    id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
	    return filtered_array

	# Run function on sample array
	binary_adaptive = filter_isolated_cells(binary_adaptive, struct=np.ones((3,3)))

	# Plot output, with all isolated single cells removed
	#plt.imshow(filtered_array, cmap=plt.cm.gray, interpolation='nearest')

	#mask[lat_start: lat_end, lon_start: lon_end] = binary_adaptive
	temp = mask[lat_start: lat_end, lon_start: lon_end]
	temp[np.where(binary_adaptive > 0)] = 1
	mask[lat_start: lat_end, lon_start: lon_end] = temp
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


#Returns list of chopped up array
#def subset_labels(img_array, lons, lats):

print("before loading teca_subtables")
#The TECA subtables are the chopped up version of the .bin file (1 TECA subtable for each day).  The subtables are in .csv form
path_to_subtables = "/global/cscratch1/sd/amahesh/segmentation_labels/teca_subtables/"
teca_subtables = [f for f in listdir(path_to_subtables) if isfile(join(path_to_subtables, f))]

print("loaded in teca_subtables")

path_to_labels = "/global/cscratch1/sd/amahesh/segmentation_labels/"
path_to_CAM5_files = "/global/cscratch1/sd/mwehner/CAM5-1-0.25degree_All-Hist_est1_v3_run2/run/h2/CAM5-1-0.25degree_All-Hist_est1_v3_run2.cam.h2."

#img_fns = [f for f in listdir(path_to_labels+"clipped_images_v2/") if isfile(join(path_to_labels+"clipped_images_v2/", f))]
#img_fns = glob.glob(path_to_labels+"clipped_images_v2/*")

progress_counter = 0
print(str(len(teca_subtables))+"LENTECASUBTABLES")

#print rank
#print(size)

for i,table_name in enumerate(teca_subtables):
	
	#UNCOMMENT THE FOLLOWING LINE IF YOU WANT TO RUN IN PARALLEL
	if i % size != rank: continue
	print(str(rank))

	if i % 2000 == 0: print(i)

	year = int(table_name[12:16])
	month = int(table_name[17:19])
	day = int(table_name[20:22])

	id_string = "{:04d}{:02d}{:02d}".format(year,month,day)

	#if len(np.flatnonzero(np.core.defchararray.find(img_fns,id_string)!=-1)) > 0:
	#	continue

	curr_table = pd.read_csv(path_to_subtables+table_name)

	#Add 3 to the storm radii
	curr_table['r0'] = curr_table['r0'][:] + 5.5

	#time_step_index refers to the 8 snapshots of data available for each data.
	for time_step_index in range(8):
		#Read in the TMQ data for the corresponding year, month, day, and time_step 
		with nc.Dataset(path_to_CAM5_files+"{:04d}-{:02d}-{:02d}-00000.nc".format(year, month, day)) as fin:
			TMQ = fin['TMQ'][:][time_step_index]
			lats = fin['lat'][:]
			lons = fin['lon'][:]
			U850 = fin.variables['U850'][:][time_step_index]
			V850 = fin.variables['V850'][:][time_step_index]
			QREFHT = fin.variables['QREFHT'][:][time_step_index]
			U850 = np.expand_dims(U850, axis=0)
			V850 = np.expand_dims(V850, axis=0)
			QREFHT = np.expand_dims(QREFHT, axis=0)

		
		#The semantic mask is one mask with 1's where the storms are and 0's everywhere else
		semantic_mask = np.zeros((image_height, image_width))
		
		#For instance segmentation, there are N masks, where N = number of instances.  Each mask is an array of size height, width
		instance_masks = []

		#For instance segmentation, the ground truth boxes surrounding the storm are also stored.  They are in the format of (N, 5) --> (x1, y1, x2, y2, classid)
		instance_boxes = []

		intersects =[]
		

		semantic_mask = get_AR_semantic_mask(U850, V850, QREFHT,time_step_index, semantic_mask)

		num_instances = 0
		num_instances = get_AR_instance_masks(U850, V850, QREFHT, time_step_index, instance_masks, instance_boxes, num_instances, lats, lons)

		#time_step_index*3 yields 0,3,6,9,12,15,18,or21
		curr_table_time = curr_table[curr_table['hour'] == time_step_index*3]
		if len(curr_table_time) >= 3:
			num_instances += len(curr_table_time)
			#ignore num_instances it is a stupid variable
			if len(curr_table_time > 0):
				for index, row in curr_table_time.iterrows():
					lat_end_index = find_nearest(lats, row['lat'] + row['r0'])
					lat_start_index = find_nearest(lats, row['lat'] - row['r0'])
					lon_end_index = find_nearest(lons, row['lon'] + row['r0'])
					lon_start_index = find_nearest(lons, row['lon'] - row['r0'])

					if len(np.unique(TMQ[lat_start_index: lat_end_index, lon_start_index: lon_end_index])) > 1:
						#Set the relevant parts of the semantic_mask to 1
						semantic_mask, intersect = binarize(TMQ, semantic_mask, lat_end_index, lat_start_index, lon_end_index, lon_start_index)
						intersects.append(intersect)

						#Create a new mask for each instance, and set the relevant parts to 1
						#instance_mask = np.zeros((image_height, image_width))
						#instance_mask, _ = binarize(TMQ, instance_mask, lat_end_index, lat_start_index, lon_end_index, lon_start_index)
						#instance_masks.append(instance_mask)

						#The ground trouth boxes are in the form x1, y1, x2, y2, class_id
						
						#instance_boxes.append(np.asarray([lat_start_index, lon_start_index, lat_end_index, lon_end_index, 1]))
						#print(instance_boxes)
				

			
			if len(instance_masks) > 0:
				#Plot sample semantic mask
				
				
				#Lats = [(-60,-36),(-36,-12),(-12,12),(12,36),(36,60)]
				lat_indices = [(0, 128, 230), (1,230, 332), (2, 332, 434), (3,435, 537), (4,537, 639)]
				intersect_flag = ""
				if np.any(np.asarray(intersects)):
					intersect_flag = "INTERSECT_"
				for lat_row_num, lat_start_index, lat_end_index in lat_indices:
					for lon_col_num, sliced_tmq in enumerate(np.split(TMQ[lat_start_index:lat_end_index,:],8,axis=1)):
						sliced_semantic_masks = np.split(semantic_mask[lat_start_index: lat_end_index,:],8,axis=1)
						#plot_mask(np.split(lons,8)[lon_col_num], lats[lat_start_index:lat_end_index], sliced_tmq, sliced_semantic_masks[lon_col_num], row['lon'], row['lat'], year, month, day, time_step_index)
						#print(sliced_semantic_masks[lon_col_num].shape)
						if np.mean(sliced_semantic_masks[lon_col_num] > 0.1) > 0.1:
							np.save(path_to_labels+"clipped_images_v4/" + intersect_flag + "{:04d}{:02d}{:02d}{:02d}-{:03d}-{:03d}-{:01d}_image.npy".format(year,month,day,time_step_index, lat_start_index, lat_end_index, lon_col_num), sliced_tmq[:,:,np.newaxis])
							np.save(path_to_labels+"semantic_combined_clipped_labels_v4/" + intersect_flag + "{:04d}{:02d}{:02d}{:02d}-{:03d}-{:03d}-{:01d}_semantic_mask.npy".format(year,month,day,time_step_index, lat_start_index, lat_end_index, lon_col_num), sliced_semantic_masks[lon_col_num])
