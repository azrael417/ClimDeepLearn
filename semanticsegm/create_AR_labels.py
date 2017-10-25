import netCDF4 as nc
import numpy as np
import datetime as dt
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
import geopy.distance
import geopy


import math
#from osgeo import ogr, osr
import matplotlib.cm as cmx
import matplotlib.colors as colors
import pylab as PP

import floodfillsearch.cFloodFillSearch as flood

def calculate_blob_length(blob):
	min_lat = lats[min(blob[0])]
	min_lon = lons[min(blob[1])]
	max_lat = lats[max(blob[0])]
	max_lon = lons[max(blob[1])]

	return geopy.distance.great_circle((max_lat, max_lon),(min_lat, min_lon)).km


filepath = "/global/cscratch1/sd/mwehner/CAM5-1-0.25degree_All-Hist_est1_v3_run2/run/h2/CAM5-1-0.25degree_All-Hist_est1_v3_run2.cam.h2.2015-12-31-00000.nc"

with nc.Dataset(filepath) as fin:
    
    TMQ = fin.variables['TMQ'][:][:][0:1, :, :]
    U850 = fin.variables['U850'][:][0:1, :, :]
    V850 = fin.variables['V850'][:][0:1, :, :]
    QREFHT = fin.variables['QREFHT'][:][0:1, :, :]
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

#Create an IVT blob array

indrand = np.random.choice(10000,len(ivt_blobs))

ivt_blob_random_array = np.zeros(IVT[0].shape)

for i, blob in zip(indrand, ivt_blobs):
    if calculate_blob_length(blob) > 1500:
    	ivt_blob_random_array[blob] = i + 1
    
ivt_blob_random_array = np.ma.masked_less_equal(ivt_blob_random_array,0)

#Plot IVT anomaly blobs for the first timestep

fig,ax = PP.subplots(figsize=(10,6))

# lon_0 is central longitude of projection.
# resolution = 'c' means use crude resolution coastlines.
m = Basemap(projection='robin',lon_0=-120,resolution='c',ax=ax)
# draw coastlines, country boundaries, fill continents.
m.drawcoastlines(linewidth=0.25)
m.drawcountries(linewidth=0.25)
# draw the edge of the map projection region (the projection limb)
m.drawmapboundary()
# draw lat/lon grid lines every 30 degrees.
m.drawmeridians(np.arange(0,360,30))
m.drawparallels(np.arange(-90,90,30))
# compute native map projection coordinates of lat/lon grid.
x, y = m(lon2d*180./np.pi, lat2d*180./np.pi)
# contour data over the map.
#cs = m.contourf(lon2d,lat2d,ivt_blob_random_array[0,...],64,latlon=True,cmap='prism')
cs = m.contourf(lon2d,lat2d,ivt_blob_random_array,64,latlon=True,cmap='prism')
#fig.colorbar(cs,ax=ax,label='IVT [kg m$^{-1}$ s$^{-1}$]')
PP.savefig("test.png")