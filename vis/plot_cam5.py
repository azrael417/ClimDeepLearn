
# coding: utf-8

# In[1]:

#get_ipython().magic('matplotlib inline')
import netCDF4 as nc
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

import datetime as dt
import pylab as PP
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap


#Load netcdf data from Michael Wehner's scratch directory
#Here, a random sample file is loaded
filepath = "/global/cscratch1/sd/mwehner/CAM5-1-0.25degree_All-Hist_est1_v3_run2/run/h2/CAM5-1-0.25degree_All-Hist_est1_v3_run2.cam.h2.2015-12-31-00000.nc"
fin = nc.Dataset(filepath)


#Extract latitude and longitude from NCfile
lats = list(fin['lat'][:])
lons = list(fin['lon'][:])
tmq = fin['TMQ'][:][0] #select the first time step

print "latitude range: " + str(min(lats)) + ", " + str(max(lats))
print "longitude range: " + str(min(lons)) + ", " + str(max(lons))

#Create a map with the Miller projection, and display the tmq data with contourf
my_map = Basemap(projection='mill', llcrnrlat=min(lats),
                 llcrnrlon=min(lons), urcrnrlat=max(lats), urcrnrlon=max(lons), resolution = 'c')
 
xx, yy = np.meshgrid(lons, lats)
x_map,y_map = my_map(xx,yy)
my_map.drawcoastlines(color="black")
my_map.contourf(x_map,y_map,tmq,64,cmap='viridis')
my_map.colorbar()
mill_proj = plt.gcf()
mill_proj.savefig("mill_projection.png")

#Clear current matplotlib plot
plt.clf()

#Create a map with the Robinson projection and display the tmq data with contourf
my_map = Basemap(projection='robin', llcrnrlat=min(lats), lon_0=180,
                 llcrnrlon=min(lons), urcrnrlat=max(lats), urcrnrlon=max(lons), resolution = 'c')
 
xx, yy = np.meshgrid(lons, lats)
x_map,y_map = my_map(xx,yy)
my_map.drawcoastlines(color="black")
my_map.contourf(x_map,y_map,tmq,64,cmap='viridis')
my_map.colorbar()
robin_proj = plt.gcf()
robin_proj.savefig("robin_projection.png")




