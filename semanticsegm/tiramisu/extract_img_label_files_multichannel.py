import matplotlib as mpl 
mpl.use('agg')
import matplotlib.pyplot as plt
import glob
import numpy as np
import os
import netCDF4 as nc
from mpi4py import MPI

#Parallelism setup
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

print("RANK: " + str(rank))
print("SIZE: " + str(size))

#Path to the clipped images, aka frames
frames_path = '/global/cscratch1/sd/amahesh/segmentation_labels/clipped_images_v4/*'

dump_path = '/global/cscratch1/sd/amahesh/segmentation_labels/dump_v7/'

#The original image height is 102 x 144.  Later, 6 pixels are clipped to make the images 96 x 144,
# which works with the DenseNet architecture
image_height = 102
image_width = 144

#Create an empty array of images.  As more channels are extracted, they are added to this array
imgs = np.zeros((25000, image_height, image_width,10))
image_metadata_new = np.zeros((25000, 7))
pointer = 0

#Set this boolean to true when you want to plot the current channels that are extracted.  Just for debugging/ vis purposes.
plot_bool = True

def plot_mask(name,img_array, year, month, day, time_step_index,lat_start_index, lat_end_index, lon_col_num):
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
        #import IPython; IPython.embed()
	plt.contourf(xx,yy,img_array.squeeze(),64,cmap='viridis')
	#plt.contourf(xx,yy,storm_mask, alpha=0.42,cmap='gray')

	mask_ex = plt.gcf()
	mask_ex.savefig(dump_path +name+"combined_mask+4{:04d}{:02d}{:02d}{:02d}-{:03d}-{:03d}-{:01d}.png".format(year,month,day,time_step_index, lat_start_index, lat_end_index, lon_col_num))
	plt.clf()

#Extract the year, month, day, time_step_index, lat_start_index, lat_end_index, and lon col number 
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

#Get the multichannel image crop based on year, month, day, time_step_index, lat_start_index, lon_col_num metadata
def get_multichannel_image_crop(year, month, day, time_step_index, lat_start_index, lat_end_index, lon_col_num):
        path_to_CAM5_files = "/global/cscratch1/sd/mwehner/CAM5-1-0.25degree_All-Hist_est1_v3_run2/run/h2/CAM5-1-0.25degree_All-Hist_est1_v3_run2.cam.h2."
        with nc.Dataset(path_to_CAM5_files+"{:04d}-{:02d}-{:02d}-00000.nc".format(year, month, day)) as fin:
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
                
                sliced_PS = np.split(PS[lat_start_index: lat_end_index,:],8,axis=1)[lon_col_num]
                sliced_PSL = np.split(PSL[lat_start_index: lat_end_index,:],8,axis=1)[lon_col_num]
                sliced_T200 = np.split(T200[lat_start_index: lat_end_index,:],8,axis=1)[lon_col_num]
                sliced_T500 = np.split(T500[lat_start_index: lat_end_index,:],8,axis=1)[lon_col_num]
                sliced_PRECT = np.split(PRECT[lat_start_index: lat_end_index,:],8,axis=1)[lon_col_num]
                sliced_TS = np.split(TS[lat_start_index: lat_end_index,:],8,axis=1)[lon_col_num]
                sliced_TREFHT = np.split(TREFHT[lat_start_index: lat_end_index,:],8,axis=1)[lon_col_num]
                sliced_U850 = np.split(U850[lat_start_index: lat_end_index,:],8,axis=1)[lon_col_num]
                sliced_V850 = np.split(V850[lat_start_index: lat_end_index,:],8,axis=1)[lon_col_num]
                sliced_QREFHT = np.split(QREFHT[lat_start_index: lat_end_index,:],8,axis=1)[lon_col_num]
                sliced_TMQ = np.split(TMQ[lat_start_index: lat_end_index,:],8,axis=1)[lon_col_num]

                return np.stack((sliced_PS,sliced_PSL,sliced_PRECT,sliced_TS,sliced_U850,sliced_V850, sliced_T200, sliced_T500, sliced_TMQ, sliced_QREFHT),axis=2)

image_metadata = np.load("/global/cscratch1/sd/amahesh/segmentation_labels/dump_v4/image_metadata.npy").astype('int')

#Extract all the channels for each image in image_metadata.npy
for i in range(image_metadata.shape[0]):
    #Parallelize the work
    if i % size != rank: continue

    #Read in the metadata from the metadata file
	year, month, day, time_step_index, lat_start_index, lat_end_index, lon_col_num = image_metadata[i]

    #Get the image crop corresponding to that metadata
    img = get_multichannel_image_crop(year, month, day, time_step_index, lat_start_index, lat_end_index, lon_col_num)
    imgs[pointer]=img
    image_metadata_new[pointer] = image_metadata[i]
    pointer += 1
    if plot_bool:
            plot_mask("PS",img[:,:,0],year, month, day, time_step_index,lat_start_index, lat_end_index, lon_col_num)
            plot_mask("PSL",img[:,:,1],year, month, day, time_step_index,lat_start_index, lat_end_index, lon_col_num)
            #plot_mask("T200",img[:,:,2],msk,year, month, day, time_step_index,lat_start_index, lat_end_index, lon_col_num)
            #plot_mask("T500",img[:,:,3],msk,year, month, day, time_step_index,lat_start_index, lat_end_index, lon_col_num)
            plot_mask("PRECT",img[:,:,2],year, month, day, time_step_index,lat_start_index, lat_end_index, lon_col_num)
            plot_mask("TS",img[:,:,3],year, month, day, time_step_index,lat_start_index, lat_end_index, lon_col_num)
            #plot_mask("TREFHT",img[:,:,6],msk,year, month, day, time_step_index,lat_start_index, lat_end_index, lon_col_num)
            plot_mask("U850",img[:,:,4],year, month, day, time_step_index,lat_start_index, lat_end_index, lon_col_num)
            plot_mask("V850",img[:,:,5],year, month, day, time_step_index,lat_start_index, lat_end_index, lon_col_num)
            #plot_mask("QREFHT",img[:,:,9],msk,year, month, day, time_step_index,lat_start_index, lat_end_index, lon_col_num)

            plot_bool = False
    print(pointer)

    #Every 80 images, save a plot of the channels that have been extracted for debugging purposes
    if pointer % 80 == 0:
            plot_bool = True
            print(pointer)

np.save(dump_path + str(rank) + 'images.npy',imgs[:pointer,:,:,:])
np.save(dump_path + str(rank)+'image_metadata.npy', image_metadata_new[:pointer])


