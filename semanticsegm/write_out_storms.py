'''
Script to write out storm with water vapor data
@Mayur
'''
import numpy as np
import pickle
import argparse
from skimage import segmentation
import netCDF4 as nc
import glob 
import matplotlib
matplotlib.use('Agg')
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
   pkl_files = glob.glob('*pkl') 
   pkl_data = pickle.load(open(pkl_files[0],'r'))
   #for each pkl file we will load the corresponding nc file
   data = nc.Dataset('/global/cscratch1/sd/mwehner/CAM5-1-0.25degree_All-Hist_est1_v3_run2/run/h2/' + pkl_files[0].split('.pkl')[0]+'.nc','r')
   #Load TMQ
   TMQ = data['TMQ']
   lat = data['lat']
   lon = data['lon']
   #Make image
   im = np.asarray(TMQ[0,:,:])
   #we'll put a dot at the right lat/lon for each image
   plt.imshow(im)
   for ii in range(len(pkl_data['lat'])):
     idx_lat = find_nearest(lat,pkl_data['lat'][0])
     idx_lon = find_nearest(lat,pkl_data['lon'][0])
     print(idx_lat,idx_lon,pkl_data['lat'][ii],pkl_data['lon'][ii])
     plt.plot(idx_lon, idx_lat, 'r*')
   plt.savefig('test.png')   
   pickle.dump(TMQ[0,:,:],open('raw.pkl','wb'))
