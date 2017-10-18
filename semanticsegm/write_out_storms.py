'''
Script to write out storm with water vapor data
@Mayur
'''
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pickle
import argparse
from skimage import segmentation
from skimage.future import graph
from skimage.filters import threshold_otsu, threshold_local, rank
from skimage.morphology import disk
import netCDF4 as nc
import glob 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


if __name__ =="__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('--savePath',default=None,type=str,help="Where the images get stored")
   parser.add_argument('--ncName',default=None,type=str,help="NetCDF4 file as input")
   parser.add_argument('--pklName',default=None,type=str,help="Name of pkl file")
   args = parser.parse_args()
   #each pkl file
   #This we get by runningt he other python script that saves storm information using a TECA query. It needs the TECA module os we have
   #separated it
   pkl_files = glob.glob('CAM*pkl') 
   pkl_data = pickle.load(open(pkl_files[0],'r'))
   #for each pkl file we will load the corresponding nc file from M Wehner's scratch
   data = nc.Dataset('/global/cscratch1/sd/mwehner/CAM5-1-0.25degree_All-Hist_est1_v3_run2/run/h2/' + pkl_files[0].split('.pkl')[0]+'.nc','r')
   #Load TMQ
   TMQ = data['TMQ']
   lat = data['lat']
   lon = data['lon']
   print lat[:], lon[:]
   #Make image
   im = np.asarray(TMQ[0,:,:])
   fig,ax = plt.subplots(1)
   #Flipping axes so we can match the lat/lon from the pkl file
   TMQ_flipped = np.flip(TMQ[0,:,:],0)
   lat_flipped = np.flip(lat,0)
   #lon_flipped = np.flip(lon,0)
   lon_flipped = lon 
   TMQ_colored = TMQ_flipped
   #When we render this, the image looks correct
   ax.imshow(TMQ_flipped, extent=(np.amin(data['lon'][:]), np.amax(data['lon'][:]), np.amin(data['lat'][:]), np.amax(data['lat'][:])))
   rect = []
   #we'll put a dot at the right lat/lon for each image
   for jj,ii in enumerate(np.unique(pkl_data['track_id'])): 
     idx = np.where(pkl_data['track_id']==ii)[0][0]
#     plt.plot(idx_lon, idx_lat, 'r*')
#     plt.figure(1)
#     plt.plot(data['lon'][:], data['lat'][:], TMQ[0,:,:])
     #This maps the lat/lon from the pkl file to the TMQ appropriately to leave the right crosses
     ax.plot(pkl_data['lon'][idx], pkl_data['lat'][idx],'r*')
     storm_radius = pkl_data['r_0'][idx]
     #We put a bounding box, we need this later for annotation
     rect.append(patches.Rectangle((pkl_data['lon'][idx]-storm_radius,pkl_data['lat'][idx]-storm_radius),2*storm_radius,2*storm_radius,linewidth=1,edgecolor='r',facecolor="none"))
     block_size = 35
     #Now we actually need indices to slice the image out and do binarization
     #This part is possibly not correct
     lon_start = pkl_data['lon'][idx]-storm_radius
     lat_start = pkl_data['lat'][idx]-storm_radius
     lon_end = pkl_data['lon'][idx]+storm_radius
     lat_end = pkl_data['lat'][idx]+storm_radius
     idx_lat_start = find_nearest(lat_flipped,lat_start)
     idx_lon_start = find_nearest(lon_flipped,lon_start)
     idx_lat_end = find_nearest(lat_flipped,lat_end)
     idx_lon_end = find_nearest(lon_flipped,lon_end)
     #im_slice = TMQ_flipped[idx_lon_end:idx_lon_start,idx_lat_end:idx_lat_start]
     im_slice = TMQ_flipped[idx_lat_end:idx_lat_start,idx_lon_start:idx_lon_end]
     #we have a try catch to do unsup binarization
     try:
       #There are four different unsup binarization methods that I am testing below
       #They all seem to perform similarly. Might need to tinker with them some more to 
       #Make them work really well. The last one is just hard coded threshold
       #adaptive_thresh = threshold_local(im_slice,block_size,offset=25)
       #adaptive_thresh = threshold_otsu(im_slice)
       #adaptive_thresh = rank.otsu((im_slice - im_slice.mean())/im_slice.max(),disk(5))
       #You need the next line for the above three methods
       #binary_adaptive = im_slice > adaptive_thresh 
       binary_adaptive = im_slice > 50.
       #TMQ_colored[idx_lon_end:idx_lon_start,idx_lat_end:idx_lat_start] = binary_adaptive
       TMQ_colored[idx_lat_end:idx_lat_start,idx_lon_start:idx_lon_end] = binary_adaptive
     except:
       print('Unable to slice')
   #This adds the bounding box the the image
   for r in rect:
     ax.add_patch(r)
   plt.savefig('test_flipped_TMQ.png')   
   plt.clf()
   #Saving images
   plt.imshow(TMQ_flipped, extent=(np.amin(data['lon'][:]), np.amax(data['lon'][:]), np.amin(data['lat'][:]), np.amax(data['lat'][:])))
   plt.savefig('raw.png')
   plt.clf()
   plt.imshow(TMQ_colored, extent=(np.amin(data['lon'][:]), np.amax(data['lon'][:]), np.amin(data['lat'][:]), np.amax(data['lat'][:])))
   plt.savefig('test_colored.png')
   pickle.dump(TMQ[0,:,:],open('raw.pkl','wb'))
