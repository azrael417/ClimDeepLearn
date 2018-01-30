import matplotlib as mpl
mpl.use('agg')

import importlib
#import utils2; importlib.reload(utils2)
from utils2 import *
import glob
import warnings
from skimage.transform import resize
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

image_height = 96 
image_width = 144

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
set_session(tf.Session(config=config))

def get_image_filename_from_mask_filename(mask_filename):
    #get just the filename of the mask.

    fname = os.path.basename(mask_filename)
    fname = fname.split("_semantic")[0]
    return fname+"_image.npy"

def _process_img_id_string(img_id_string):
  year = int(img_id_string[:4])
  month = int(img_id_string[4:6])
  day = int(img_id_string[6:8])
  time_step = int(img_id_string[8:10])
  return year, month, day, time_step

warnings.filterwarnings("ignore",category=DeprecationWarning)

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




#TESTING THE TIRAMISU model#


#Load the images and the labels
imgs = np.load("/global/cscratch1/sd/tkurth/gb2018/climate/tiramisu_clipped_combined_v2/images.npy")
imgs = imgs.reshape([imgs.shape[0],imgs.shape[1],imgs.shape[2],1])
labels = np.load("/global/cscratch1/sd/tkurth/gb2018/climate/tiramisu_clipped_combined_v2/masks.npy")

#Image metadata contains year, month, day, time_step, and lat/ lon data for each crop.  
#See README in $SCRATCH/segmentation_labels/dump_v4 on CORI
image_metadata = np.load("/global/cscratch1/sd/tkurth/gb2018/climate/tiramisu_clipped_combined_v2/image_metadata.npy")

imgs = imgs[:,3:-3,...]
labels = labels[:,3:-3,:]

#PERMUTATION OF DATA
np.random.seed(12345)
shuffle_indices = np.random.permutation(len(imgs)) 
np.save("./shuffle_indices.npy", shuffle_indices)
imgs = imgs[shuffle_indices]
labels = labels[shuffle_indices]
image_metadata = image_metadata[shuffle_indices]

#Create train/validation/test split
trn = imgs[:int(0.8*len(imgs))]
trn_labels = labels[:int(0.8 * len(labels))]
test = imgs[int(0.8*len(imgs)):int(0.9*len(imgs))]
test_labels = labels[int(0.8*len(imgs)):int(0.9*len(imgs))]
valid = imgs[int(0.9*len(imgs)):]
valid_labels = labels[int(0.9*len(imgs)):]

rnd_trn = len(trn_labels)
rnd_test = len(test_labels)


 ## Train

#limit_mem()

input_shape = (imgs.shape[1],imgs.shape[2],1)

img_input = Input(shape=input_shape)
#x = create_tiramisu(3, img_input, nb_layers_per_block=[4,5,7,10,12,15], p=0.2, wd=1e-4)
blocks=[3,3,4,7,10]
x = create_tiramisu(3, img_input,nb_layers_per_block=blocks, p=0.2, wd=1e-4)
model = Model(img_input, x)

#This is an attempt to weight classes by frequency...

# class_weight = class_weight.compute_class_weight('balanced', np.unique(trn_labels), trn_labels.flatten())
# #import IPython; IPython.embed()
# class_weight = dict(enumerate(class_weight))

# sample_weights = np.zeros((trn_labels.shape[0], image_height*image_width))
# trn_labels_flat = trn_labels.reshape((trn_labels.shape[0],image_height*image_width))
# sample_weights[np.where(trn_labels_flat == 0)] = class_weight[0]
# sample_weights[np.where(trn_labels_flat == 1)] = class_weight[1]
# sample_weights[np.where(trn_labels_flat == 2)] = class_weight[2]


# model.compile(loss='sparse_categorical_crossentropy', 
#               optimizer=keras.optimizers.RMSprop(1e-3), metrics=["accuracy"], sample_weight = sample_weights,sample_weight_mode = "temporal")

#Create a checkpoint callback
cp_callback = keras.callbacks.ModelCheckpoint("./training_round3/weights-{epoch:02d}-{val_loss:.2f}-" + str(len(blocks)) +"-" + str(time.time())[-7:-3] + "-.hdf5", monitor='val_loss', verbose=2, save_best_only=False, save_weights_only=False, mode='auto', period=1)

model.compile(loss='sparse_categorical_crossentropy', 
               optimizer=keras.optimizers.RMSprop(1e-3), metrics=["accuracy"])

model.fit(trn, trn_labels.squeeze().reshape(trn.shape[0],image_height*image_width,1), 
    nb_epoch=300,shuffle=True, verbose=2, validation_data=(valid, valid_labels.squeeze().reshape(valid.shape[0],image_height*image_width,1)),
    callbacks=[cp_callback])


