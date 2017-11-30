import matplotlib as mpl 
mpl.use('agg')
import matplotlib.pyplot as plt
import glob
import numpy as np
import os


frames_path = '/global/cscratch1/sd/amahesh/segmentation_labels/clipped_images_v4/*'

#labels_path = '/home/mudigonda/Data/tiramisu_labels/[0-9]*'
labels_path = '/global/cscratch1/sd/amahesh/segmentation_labels/semantic_combined_clipped_labels_v4/*'

dump_path = '/global/cscratch1/sd/amahesh/segmentation_labels/dump_v4/'

fnames = glob.glob(frames_path+"*")
fnames.sort()
mask_filenames = glob.glob(labels_path+"*")

image_height = 102
image_width = 144

imgs = np.zeros((len(fnames),image_height,image_width))
masks = np.zeros((len(fnames), image_height, image_width))
image_metadata = np.zeros((len(fnames), 7))
pointer = 0

plot_bool = True

#def plot_mask(lons, lats, img_array, storm_mask, year, month, day, time_step_index,lat_row_num, lon_col_num):
def plot_mask(img_array, storm_mask, year, month, day, time_step_index,lat_start_index, lat_end_index, lon_col_num):
	#my_map = Basemap(projection='robin', llcrnrlat=min(lats), lon_0=np.median(lons),
    #              llcrnrlon=min(lons), urcrnrlat=max(lats), urcrnrlon=max(lons), resolution = 'c')
 
	xx, yy = np.meshgrid(np.arange(0,144), np.arange(0,102))
	# x_map,y_map = my_map(xx,yy)
	# my_map.drawcoastlines(color="black")
	# my_map.contourf(x_map,y_map,img_array,64,cmap='viridis')
	# cbar = my_map.colorbar()
	# my_map.contourf(x_map,y_map,storm_mask, alpha=0.42,cmap='gray')
	# my_map.drawmeridians(np.arange(-180, 180, 60), labels=[0,0,0,1])
	# my_map.drawparallels(np.arange(-90, 90, 30), labels =[1,0,0,0])
	# plt.title("TMQ with Segmented TECA Storms")
	# cbar.ax.set_ylabel('TMQ kg $m^{-2}$')
	plt.contourf(xx,yy,img_array.squeeze(),64,cmap='viridis')
	plt.contourf(xx,yy,storm_mask, alpha=0.42,cmap='gray')

	mask_ex = plt.gcf()
	mask_ex.savefig(dump_path +"combined_mask+4{:04d}{:02d}{:02d}{:02d}-{:03d}-{:03d}-{:01d}.png".format(year,month,day,time_step_index, lat_start_index, lat_end_index, lon_col_num))
	plt.clf()

def process_image_name(image_name):
	image_name = os.path.basename(image_name)
	if "INTERSECT_" in image_name:
		string_id = image_name.split("_")[1]
	else:
		string_id = image_name.split("_")[0]
	year = string_id[0:4]
	month = string_id[4:6]
	day = string_id[6:8]
	time_step_index = string_id[8:10]
	lat_start_index = string_id[11:14]
	lat_end_index = string_id[15:18]
	lon_col_num = string_id[19]
	#INTERSECT_2015101404-537-639-7_semantic_mask.npy
	return int(year), int(month), int(day), int(time_step_index),int(lat_start_index), int(lat_end_index),int(lon_col_num)

def get_mask_filename_from_image_filename(image_filename):
    #get just the filename of the image, not the whole path to it.
    fname = os.path.basename(image_filename)
    fname = fname.split("_image")[0]
    return labels_path[:len(labels_path)-1]+fname+"_semantic_mask.npy"

for i,image_name in enumerate(fnames):
	mask_name = get_mask_filename_from_image_filename(image_name)
	try:
		msk = np.load(mask_name)
		if np.mean(msk > 0.1) > 0.1:
			img = np.load (image_name)
			imgs[pointer] = img.squeeze()
			masks[pointer] = msk
			year, month, day, time_step_index, lat_start_index, lat_end_index, lon_col_num = process_image_name(image_name)
			image_metadata[pointer] = [year, month, day, time_step_index, lat_start_index, lat_end_index, lon_col_num]
			pointer += 1
			if plot_bool:	
				plot_mask(img,msk,year, month, day, time_step_index,lat_start_index, lat_end_index, lon_col_num)
				plot_bool = False
		if i % 8000 == 0:
			plot_bool = True
			print(i)
			print("pointer: " + str(pointer))
	except:
		print("couldn't read " + mask_name)

np.save(dump_path + 'images.npy',imgs[:pointer,:,:])
np.save(dump_path + 'masks.npy',masks[:pointer,:,:])
np.save(dump_path + 'image_metadata.npy', image_metadata[:pointer])


