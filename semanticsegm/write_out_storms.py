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
import netCDF4 as nc
import glob 
import matplotlib.pyplot as plt

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
   pkl_files = glob.glob('CAM*pkl') 
   pkl_data = pickle.load(open(pkl_files[0],'r'))
   #for each pkl file we will load the corresponding nc file
   data = nc.Dataset('/global/cscratch1/sd/mwehner/CAM5-1-0.25degree_All-Hist_est1_v3_run2/run/h2/' + pkl_files[0].split('.pkl')[0]+'.nc','r')
   #Load TMQ
   TMQ = data['TMQ']
   lat = data['lat']
   lon = data['lon']
   print lat[:], lon[:]
   #Make image
   im = np.asarray(TMQ[0,:,:])
   #we'll put a dot at the right lat/lon for each image
   f1 = plt.figure(1)
#   plt.imshow(im)
   for ii in np.unique(pkl_data['track_id']): 
     idx = np.where(pkl_data['track_id']==ii)[0][0]
     idx_lat = find_nearest(lat,pkl_data['lat'][idx])
     idx_lon = find_nearest(lat,pkl_data['lon'][idx])
     print "pkl_data[lat] is:"
     print pkl_data['lat'][:]
     print "pkl_data[lat][idx] is:" 
     print pkl_data['lat'][idx]
     print "pkl_data[lon] is:"
     print pkl_data['lon'][:]
     print "pkl_data[lon][idx] is:"
     print pkl_data['lon'][idx]
     print "idx_lat and idx_lon are:"
     print idx_lat, idx_lon
     idx_rad_lat_min = find_nearest(lat,pkl_data['lat'][idx]+pkl_data['r_0'][idx])
     idx_rad_lon_min = find_nearest(lon,pkl_data['lon'][idx]+pkl_data['r_0'][idx])
     idx_rad_lat_max = find_nearest(lat,pkl_data['lat'][idx]-pkl_data['r_0'][idx])
     idx_rad_lon_max = find_nearest(lon,pkl_data['lon'][idx]-pkl_data['r_0'][idx])
     print(idx_lat,idx_rad_lat_min,idx_rad_lat_max,idx_lon,idx_rad_lon_min,idx_rad_lon_max)
     f2 = plt.figure(2)
     im_slice = im[idx_rad_lon_max:idx_rad_lon_min,idx_rad_lat_max:idx_rad_lat_min]
     #im_slice = im[idx_rad_lat_max:idx_rad_lon_min,idx_rad_lat_max:idx_rad_lat_min]
     try:
         labels = segmentation.slic((im_slice-im_slice.mean())/im_slice.max(),compactness=0.01)
         plt.imshow(labels)
         plt.savefig('Subim-'+str(ii)+'.png')
         plt.close()
     except:
         print("Couldn't do it")
     plt.figure(1)
#     plt.plot(idx_lon, idx_lat, 'r*')
#     plt.figure(1)
#     plt.plot(data['lon'][:], data['lat'][:], TMQ[0,:,:])
     TMQ_flipped = np.flip(TMQ[0,:,:],0)
     plt.imshow(TMQ_flipped, extent=(np.amin(data['lon'][:]), np.amax(data['lon'][:]), np.amin(data['lat'][:]), np.amax(data['lat'][:])))
     plt.plot(pkl_data['lon'][idx], pkl_data['lat'][idx],'r*')
   plt.savefig('test_flipped_TMQ.png')   
   plt.clf()
   plt.imshow(im)
   plt.savefig('raw.png')
   pickle.dump(TMQ[0,:,:],open('raw.pkl','wb'))
