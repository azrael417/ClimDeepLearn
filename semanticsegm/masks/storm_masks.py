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

filepath = "/global/cscratch1/sd/mwehner/CAM5-1-0.25degree_All-Hist_est1_v3_run2/run/h2/CAM5-1-0.25degree_All-Hist_est1_v3_run2.cam.h2.2015-12-31-00000.nc"

fin = nc.Dataset(filepath)
lats = fin['lat'][:]
lons = fin['lon'][:]
tmq = fin['TMQ'][:][0] #select the first time step

print "latitude range: " + str(min(lats)) + ", " + str(max(lats))
print "longitude range: " + str(min(lons)) + ", " + str(max(lons))

my_map = Basemap(projection='robin', llcrnrlat=min(lats), lon_0=180,
                  llcrnrlon=min(lons), urcrnrlat=max(lats), urcrnrlon=max(lons), resolution = 'c')
 
xx, yy = np.meshgrid(lons, lats)
x_map,y_map = my_map(xx,yy)
storm_lon = 66
storm_lat = 17
x_plot, y_plot = my_map(storm_lon, storm_lat)
my_map.drawcoastlines(color="black")
my_map.contourf(x_map,y_map,tmq,64,cmap='viridis')
my_map.plot(x_plot, y_plot, 'r*', color = "red")
my_map.colorbar()

plot_point_ex = plt.gcf()
plot_point_ex.savefig("plot_point_ex.png")

#Clear current matplotlib plot
plt.clf()

#plt.figure(figsize=(40,20))
#plt.show()

def get_grid_cell_indices(given_lat,given_lon,lat_grid,lon_grid):
    distances = np.sqrt((given_lon - lon_grid)**2 + (given_lat - lat_grid)**2)
    #return lat_grid[np.where(distances == np.amin(distances))][0], lon_grid[np.where(distances == np.amin(distances))][0]
    return np.where(distances == np.amin(distances))[0][0], np.where(distances == np.amin(distances))[1][0]

# Define starting point.
start = geopy.Point(storm_lat, storm_lon)


# Define a general distance object, initialized with a distance of 400 km.
radius = 800
d = geopy.distance.VincentyDistance(kilometers = radius)

# Use the `destination` method with a bearing of 0 degrees (which is north)
# in order to go from point `start` 1 km to north.
north_point = d.destination(point=start, bearing=0)
south_point = d.destination(point=start, bearing=180)
east_point = d.destination(point=start, bearing=90)
west_point = d.destination(point=start, bearing=270)

urcrnrlat = north_point.latitude
urcrnrlon = east_point.longitude
llcrnrlat = south_point.latitude
llcrnrlon = west_point.longitude

urcrnr_grid_indices = get_grid_cell_indices(urcrnrlat, urcrnrlon, yy, xx)
llcrnr_grid_indices = get_grid_cell_indices(llcrnrlat, llcrnrlon, yy, xx)

i = llcrnr_grid_indices[0]
j = llcrnr_grid_indices[1]

storm_mask = np.zeros((768,1152))

while i < urcrnr_grid_indices[0]:
	while j < urcrnr_grid_indices[1]:
		if geopy.distance.vincenty((yy[i,j],xx[i,j]), (storm_lat,storm_lon)).kilometers < radius:
			storm_mask[i][j] = 1
		j += 1
	j = llcrnr_grid_indices[1]
	i += 1

print np.sum(storm_mask > 0)

my_map = Basemap(projection='robin', llcrnrlat=min(lats), lon_0=180,
                  llcrnrlon=min(lons), urcrnrlat=max(lats), urcrnrlon=max(lons), resolution = 'c')
 
xx, yy = np.meshgrid(lons, lats)
x_map,y_map = my_map(xx,yy)
storm_lon = 66
storm_lat = 17
x_plot, y_plot = my_map(storm_lon, storm_lat)
my_map.drawcoastlines(color="black")
my_map.contourf(x_map,y_map,tmq,64,cmap='viridis')
#my_map.plot(x_plot, y_plot, 'r*', color = "red")
my_map.colorbar()
my_map.contourf(x_map,y_map,storm_mask, alpha=0.2,cmap='gray')

mask_ex = plt.gcf()
mask_ex.savefig("mask_ex.png")
