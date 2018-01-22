import matplotlib as mpl
mpl.use('agg')

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import importlib
#import utils2; importlib.reload(utils2)
from utils2 import *
import glob
import warnings
#from skimage.transform import resize
import time
import matplotlib.pyplot as plt

#from mpl_toolkits.basemap import Basemap
import numpy as np
import argparse
import netCDF4 as nc
from os import listdir
from os.path import isfile, join
import pandas as pd
import pickle
from sklearn.utils import class_weight

import subprocess as sp
import os

#scale_down = 8
#image_height = 768 / scale_down
#image_width = 1152 / scale_down

image_height = 96 
image_width = 144

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import RandomSearchCV

#Get list of available GPUs
# ACCEPTABLE_AVAILABLE_MEMORY = 1024
# COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"

# _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
# memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
# memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
# available_gpus = [i for i, x in enumerate(memory_free_values) if x > ACCEPTABLE_AVAILABLE_MEMORY]


#set the gpu for the job to be the available GPU
#config = tf.ConfigProto(device_count = {'GPU': available_gpus[0]})
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
set_session(tf.Session(config=config))

def get_image_filename_from_mask_filename(mask_filename):
    #get just the filename of the mask.

    fname = os.path.basename(mask_filename)
    fname = fname.split("_semantic")[0]
    return fname+"_image.npy"

def downsize_labels(lbl_array, image_height, image_width):
    temp_1 = np.copy(lbl_array)
    temp_2 = np.copy(lbl_array)

    temp_1[np.where(temp_1) != 1] = 0
    temp_2[np.where(temp_2) != 2] = 0
    temp_1 = resize(temp_1, (image_height, image_width))
    temp_2 = resize(temp_2, (image_height, image_width))
    new_array = np.zeros((image_height, image_width))
    new_array[np.where(temp_1) > 0] = 1
    new_array[np.where(temp_2) > 0] = 2
    return new_array

def plot_mask(lons, lats, img_array, storm_mask):
    # my_map = Basemap(projection='robin', llcrnrlat=min(lats), lon_0=180,
    #               llcrnrlon=min(lons), urcrnrlat=max(lats), urcrnrlon=max(lons), resolution = 'c')
 
    xx, yy = np.meshgrid(lons, lats)
    # x_map,y_map = my_map(xx,yy)
    # x_plot, y_plot = my_map(storm_lon, storm_lat)
    # my_map.drawcoastlines(color="black")
    plt.contourf(xx,yy,img_array,64,cmap='viridis')
    #my_map.plot(x_plot, y_plot, 'r*', color = "red")
    #cbar = my_map.colorbar()
    #plt.contourf(xx,yy,storm_mask, alpha=0.42,cmap='gray')
    plt.title("TMQ with Segmented TECA Storms")
    #cbar.ax.set_ylabel('TMQ kg $m^{-2}$')

    mask_ex = plt.gcf()
    mask_ex.savefig("test.png")
    plt.clf()


def load_lat_lon():
    #sample filepath
    filepath = "/home/mudigonda/files_for_first_maskrcnn_test/CAM5-1-0.25degree_All-Hist_est1_v3_run2.cam.h2.2013-01-11-00000.nc"
    print(filepath)
    with nc.Dataset(filepath) as fin:
        TMQ = fin['TMQ'][:][time_step]
        lons = fin['lon'][:]
        lats = fin['lat'][:]
        print(TMQ.shape)
    return lons[::scale_down], lats[::scale_down], (TMQ * 1000).astype('uint32')


def _process_img_id_string(img_id_string):
  year = int(img_id_string[:4])
  month = int(img_id_string[4:6])
  day = int(img_id_string[6:8])
  time_step = int(img_id_string[8:10])
  return year, month, day, time_step

warnings.filterwarnings("ignore",category=DeprecationWarning)

#frames_path = '/home/mudigonda/Data/tiramisu_images/'
#frames_path = '/home/mudigonda/Data/tiramisu_clipped_images/*'

#labels_path = '/home/mudigonda/Data/tiramisu_labels/[0-9]*'
#labels_path = '/home/mudigonda/Data/tiramisu_clipped_labels/*'

# fnames = glob.glob(frames_path+"*")
# mask_filenames = glob.glob(labels_path+"*")

# #imgs = np.stack([np.load(fn) for fn in large_percent_image_filenames])
# print(imgs.dtype)

# #Uncomment if you want to scale down images
# #imgs = np.asarray([resize(img, (image_height, image_width),preserve_range=True)for img in imgs]).astype('uint32')

# #labels = np.stack([np.load(fn) for fn in lnames]).astype('uint8')
# #labels = np.asarray([downsize_labels(lbl, image_height, image_width)for lbl in labels])
# #labels = np.stack([np.load(fn) for fn in large_percent_mask_filenames])

# #Uncomment if you want to scale down labels
# #labels = labels[:,::scale_down,::scale_down]
# print((imgs.shape,labels.shape))


# TMQ = imgs[len(imgs)-1]
# label = labels[len(labels) - 1]
# id_start_index = min([i for i, c in enumerate(fnames[len(fnames) - 5]) if c.isdigit()])
# img_id = fnames[len(fnames) - 5][id_start_index:]
# print(img_id)
# year, month, day, time_step = _process_img_id_string(img_id)
# tim
# plot_mask(lons, lats, TMQ.squeeze(), label)

def relu(x): return Activation('relu')(x)
def dropout(x, p): return Dropout(p)(x) if p else x
def bn(x): return BatchNormalization(mode=2, axis=-1)(x)
def relu_bn(x): return relu(bn(x))
def concat(xs): return merge(xs, mode='concat', concat_axis=-1)

def conv(x, nf, sz, wd, p, stride=1): 
    x = Convolution2D(nf, sz, sz, init='he_uniform', border_mode='same', 
                      subsample=(stride,stride), W_regularizer=l2(wd))(x)
    return dropout(x, p)

def conv_relu_bn(x, nf, sz=3, wd=0, p=0, stride=1): 
    return conv(relu_bn(x), nf, sz, wd=wd, p=p, stride=stride)

def dense_block(n,x,growth_rate,p,wd):
    added = []
    for i in range(n):
        b = conv_relu_bn(x, growth_rate, p=p, wd=wd)
        x = concat([x, b])
        added.append(b)
    return x,added

# This is the downsampling transition. 
# 
# In the original paper, downsampling consists of 1x1 convolution followed by max pooling. However we've found a stride 2 1x1 convolution to give better results.


def transition_dn(x, p, wd):
#     x = conv_relu_bn(x, x.get_shape().as_list()[-1], sz=1, p=p, wd=wd)
#     return MaxPooling2D(strides=(2, 2))(x)
    return conv_relu_bn(x, x.get_shape().as_list()[-1], sz=1, p=p, wd=wd, stride=2)

# Next we build the entire downward path, keeping track of Dense block outputs in a list called `skip`. 

def down_path(x, nb_layers, growth_rate, p, wd):
    skips = []
    for i,n in enumerate(nb_layers):
        x,added = dense_block(n,x,growth_rate,p,wd)
        skips.append(x)
        x = transition_dn(x, p=p, wd=wd)
    return skips, added

# This is the upsampling transition. We use a deconvolution layer.

def transition_up(added, wd=0):
    x = concat(added)
    _,r,c,ch = x.get_shape().as_list()
    return Deconvolution2D(ch, 3, 3, (None,r*2,c*2,ch), init='he_uniform', 
               border_mode='same', subsample=(2,2), W_regularizer=l2(wd))(x)
#     x = UpSampling2D()(x)
#     return conv(x, ch, 2, wd, 0)

# This builds our upward path, concatenating the skip connections from `skip` to the Dense block inputs as mentioned.

def up_path(added, skips, nb_layers, growth_rate, p, wd):
    for i,n in enumerate(nb_layers):
        x = transition_up(added, wd)
        x = concat([x,skips[i]])
        x,added = dense_block(n,x,growth_rate,p,wd)
    return x


# ### Build the tiramisu model

# - nb_classes: number of classes
# - img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
# - depth: number or layers
# - nb_dense_block: number of dense blocks to add to end (generally = 3)
# - growth_rate: number of filters to add per dense block
# - nb_filter:  initial number of filters
# - nb_layers_per_block: number of layers in each dense block.
#   - If positive integer, a set number of layers per dense block.
#   - If list, nb_layer is used as provided
# - p: dropout rate
# - wd: weight decay

def reverse(a): return list(reversed(a))

# Finally we put together the entire network.

def create_tiramisu(nb_classes, img_input, nb_dense_block=6, 
    growth_rate=16, nb_filter=48, nb_layers_per_block=5, p=None, wd=0):
    
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)
    else: nb_layers = [nb_layers_per_block] * nb_dense_block

    x = conv(img_input, nb_filter, sz=3, wd=wd, p=0)
    skips,added = down_path(x, nb_layers, growth_rate, p, wd)
    x = up_path(added, reverse(skips[:-1]), reverse(nb_layers[:-1]), growth_rate, p, wd)
    
    x = conv(x, nb_classes, 1, wd, 0)
    _,r,c,f = x.get_shape().as_list()
    x = Reshape((-1, nb_classes))(x)
    return Activation('softmax')(x)

    #growth_rate = params['growth_rate']

imgs = np.load("/global/cscratch1/sd/amahesh/segmentation_labels/dump_v4/images.npy")
imgs = imgs.reshape([imgs.shape[0],imgs.shape[1],imgs.shape[2],1])
labels = np.load("/global/cscratch1/sd/amahesh/segmentation_labels/dump_v4/masks.npy")

#Image metadata contains year, month, day, time_step, and lat/ lon data for each crop
image_metadata = np.load("/global/cscratch1/sd/amahesh/segmentation_labels/dump_v4/image_metadata.npy")

imgs = imgs[:,3:-3,...]
labels = labels[:,3:-3,:]

#PERMUTATION OF DATA
np.random.seed(12345)
shuffle_indices = np.random.permutation(len(imgs)) 
np.save("./shuffle_indices.npy", shuffle_indices)
imgs = imgs[shuffle_indices]
labels = labels[shuffle_indices]
image_metadata = image_metadata[shuffle_indices]

trn = imgs[:int(0.8*len(imgs))]
trn_labels = labels[:int(0.8 * len(labels))]
test = imgs[int(0.8*len(imgs)):int(0.9*len(imgs))]
test_labels = labels[int(0.8*len(imgs)):int(0.9*len(imgs))]
valid = imgs[int(0.9*len(imgs)):]
valid_labels = labels[int(0.9*len(imgs)):]

rnd_trn = len(trn_labels)
rnd_test = len(test_labels)

def build_model(block1, block2, block3, block4, block5, learning_rate=1e-4, dropout_prob=0.2, weight_decay=0):
    input_shape = (imgs.shape[1],imgs.shape[2],1)

    img_input = Input(shape=input_shape)    
    blocks=[block1, block2, block3, block4, block5]

    x = create_tiramisu(3, img_input,nb_layers_per_block=blocks, p=dropout_prob, wd=weight_decay, growth_rate=16)
    model = Model(img_input, x)

    model.compile(loss='sparse_categorical_crossentropy', 
                   optimizer=keras.optimizers.RMSprop(learning_rate), metrics=["accuracy"])


batch_size = [8, 16, 32, 62, 128]

cp_callback = keras.callbacks.ModelCheckpoint("./sklearn_training/weights-{epoch:02d}-{val_loss:.2f}-" + "5" +"-" + str(time.time())[-7:-3] + "-.hdf5", monitor='val_loss', verbose=2, save_best_only=False, save_weights_only=False, mode='auto', period=1)

early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              verbose=0, mode='auto')

model = KerasRegressor(build_fn=build_model, batch_size=batch_size,
        nb_epoch=1,shuffle=True, verbose=2, validation_data=(valid, valid_labels.squeeze().reshape(valid.shape[0],image_height*image_width,1)),
        callbacks=[cp_callback, early_stopping_callback])
block1 = [4, 8, 16, 32, 64, 128]
block2 = [4, 8, 16, 32, 64, 128]
block3 = [4, 8, 16, 32, 64, 128]
block4 = [4, 8, 16, 32, 64, 128]
block5 = [4, 8, 16, 32, 64, 128]
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
weight_decay = [0, 0.1, 0.3, 0.5]
dropout_prob = [0.2, 0.35, 0.5]
param_grid = dict(batch_size=batch_size, block1=block1, block2=block2, block3=block3, 
    block4=block4, block5=block5, learning_rate=learning_rate, weight_decay=weight_decay,
    dropout_prob=dropout_prob)
grid = GridSearchCV(estimator=model, param_grid=param_grid,n_jobs=-1)

grid.fit(trn, trn_labels.squeeze().reshape(trn.shape[0],image_height*image_width,1))

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

