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

#unsupervised technique - binarize a portion of the image by setting the mask to either 1 or 0
def binarize(img_array, mask, lat_end, lat_start, lon_end, lon_start):
	im_slice = img_array[lat_start: lat_end, lon_start: lon_end]
	#adaptive_thresh = threshold_local(im_slice,block_size,offset=25)
    #adaptive_thresh = rank.otsu((im_slice - im_slice.mean())/im_slice.max(),disk(5))
	adaptive_thresh = threshold_otsu(im_slice)
	binary_adaptive = im_slice > adaptive_thresh 
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
	my_map.colorbar()
	my_map.contourf(x_map,y_map,storm_mask, alpha=0.42,cmap='gray')

	mask_ex = plt.gcf()
	mask_ex.savefig("./sample_seg_masks/teca_storm_mask{:04d}-{:02d}-{:02d}-{:02d}.png".format(year,month,day,time_step_index))
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
	#curr_table['r0'] = curr_table['r0'][:] + 3

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
		

		#time_step_index*3 yields 0,3,6,9,12,15,18,or21
		curr_table_time = curr_table[curr_table['hour'] == time_step_index*3]
		num_instances = len(curr_table_time)
		if len(curr_table_time > 0):
			for index, row in curr_table_time.iterrows():
				#Boolean determining whether or not to calculate the threshold of this storm.  
				#If the storm is too small (and doesn't span multiple pixels), don't calculate its threshold
				calc_threshold = False
				lat_end_index = find_nearest(lats, row['lat'] + row['r0'])
				lat_start_index = find_nearest(lats, row['lat'] - row['r0'])
				lon_end_index = find_nearest(lons, row['lon'] + row['r0'])
				lon_start_index = find_nearest(lons, row['lon'] - row['r0'])

				if len(np.unique(TMQ[lat_start_index: lat_end_index, lon_start_index: lon_end_index])) > 1:
					calc_threshold = True

				
				if calc_threshold:
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

				np.save(path_to_labels+"semantic_storm_labels/{:04d}-{:02d}-{:02d}-{:02d}.npy".format(year,month,day,time_step_index), semantic_mask)
				instance_labels = [num_instances,np.asarray(instance_boxes), np.asarray(instance_masks)]
				with open(path_to_labels+"instance_storm_labels/{:04d}-{:02d}-{:02d}-{:02d}.pkl".format(year,month,day,time_step_index),'w') as f:
					pickle.dump(instance_labels,f)