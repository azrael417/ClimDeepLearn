import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import h5py as h5
from mpl_toolkits.basemap import Basemap
import numpy as np

from matplotlib.colors import ListedColormap
import sys
import os

# Choose colormap
cmap = mpl.cm.viridis

# Get the colormap colors
my_cmap = cmap(np.arange(cmap.N))

# Set alpha
alpha = np.linspace(0, 1, cmap.N)
my_cmap[:,0] = (1-alpha) + alpha * my_cmap[:,0]
my_cmap[:,1] = (1-alpha) + alpha * my_cmap[:,1]
my_cmap[:,2] = (1-alpha) + alpha * my_cmap[:,2]
#my_cmap[:,0] = np.maximum(my_cmap[:,0], my_cmap[:,2])
#my_cmap[:,1] = np.maximum(my_cmap[:,1], my_cmap[:,2])
#my_cmap[:,-1] = np.linspace(0, 1, cmap.N)

# Create new colormap
my_cmap = ListedColormap(my_cmap)

#print l.shape
#print l1
#exit(1)

dpath = '.'
dpath = '../dset/val'

p = np.load(sys.argv[1])
if len(sys.argv) > 2:
    dfile = sys.argv[2]
else:
    dfile = str(p['filename'])
    if dpath is not None:
        dfile = os.path.join(dpath, os.path.basename(dfile))
print dfile
l = p['label'] / 100
p = p['prediction']
p = np.roll(p,[0,1152/2])
p1 = (p == 100)
p2 = (p == 200)

h = h5.File(dfile)
d = h['climate']['data'][0,...]
d = np.roll(d,[0,1152/2])

#l = h['climate']['labels']
l_ref = h['climate']['labels']
errors = np.amax(np.abs(l - l_ref))
print '{} errors'.format(errors)
l = np.roll(l,[0,1152/2])
l1 = (l == 1)
l2 = (l == 2)

lats = np.linspace(-90,90,768)
longs = np.linspace(-180,180,1152)
#print longs[0:10], longs[-10:]
#exit(1)

def do_fig(full_size, figsize, fname):
    fig = plt.figure(figsize=figsize)

    if full_size:
        my_map = Basemap(projection='eck4', lon_0=0,
                         resolution = 'c')
        xx, yy = np.meshgrid(longs, lats)
        x_map,y_map = my_map(xx,yy)
        my_map.drawcoastlines(color=[0.5,0.5,0.5])
        my_map.contourf(x_map,y_map,d,64,cmap=my_cmap)
        cbar = my_map.colorbar()
        if True:
            my_map.contour(x_map,y_map,p1,[0.5],linewidths=1,colors='red')
            my_map.contour(x_map,y_map,p2,[0.5],linewidths=1,colors='blue')
        if False:
            my_map.contour(x_map,y_map,l1,[0.5],linewidths=1,colors='pink')
            my_map.contour(x_map,y_map,l2,[0.5],linewidths=1,colors='cyan')

        my_map.drawmeridians(np.arange(-180, 180, 60), labels=[0,0,0,1])
        my_map.drawparallels(np.arange(-90, 90, 30), labels =[1,0,0,0])
    else:
        my_map = Basemap(projection='merc', #lon_0=120, lat_0=0,
                         llcrnrlat=-21, llcrnrlon=95,
                         urcrnrlat=41, urcrnrlon=165,
                         resolution = 'l')
        #my_map = Basemap(projection='eck4', lon_0=0,
        #                 resolution = 'c')
        xx, yy = np.meshgrid(longs, lats)
        x_map,y_map = my_map(xx,yy)
        my_map.drawcoastlines(color=[0.7,0.7,0.7])
        my_map.fillcontinents(color=[0.9,0.9,0.9],lake_color="white")
        my_map.contourf(x_map,y_map,p1,[0.5,1.5],colors=['red'], zorder=10)
        my_map.contourf(x_map,y_map,p2,[0.5,1.5],colors=[[0.5,0.5,1]], zorder=10)
        my_map.contour(x_map,y_map,l1,[0.5],linewidths=1,colors='black', zorder=20,antialiased=True)
        my_map.contour(x_map,y_map,l2,[0.5],linewidths=1,colors='black', zorder=20,antialiased=True)

        my_map.drawmeridians(np.arange(-180, 180, 30), labels=[0,0,0,1])
        my_map.drawparallels(np.arange(-90, 90, 15), labels =[1,0,0,0])
 


    mask_ex = plt.gcf()
    #mask_ex.savefig("/global/cscratch1/sd/amahesh/segm_plots/combined_mask+{:04d}-{:02d}-{:02d}-{:02d}-{:02d}.png".format(year,month,day,time_step_index, run_num))
    #mask_ex.savefig("/global/cscratch1/sd/mayur/segm_plots/combined_mask+{:04d}-{:02d}-{:02d}-{:02d}-{:02d}.png".format(year,month,day,time_step_index, run_num))
    mask_ex.savefig(fname,bbox_inches='tight')
    plt.clf()

do_fig(True, (10,7), 'results_full.png')
do_fig(False, (5.5,5.5), 'results_zoom.png')
