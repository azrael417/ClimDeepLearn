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

#Set up tf session. Let tensorflow grab all available memory at the beginning
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
set_session(tf.Session(config=config))

num_cropped_pixels = 6

from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model

import tensorflow as tf

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

#------------------------------------- Tiramisu Architecture Setup -------------------------------#

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


# -------------------------------------------------- Generator to Load in Multi-Channel Data --------------------#

def multi_channel_generator(batch_size,training_ranks):
    dataset_path = "/home/mudigonda/Data/multi_channel_v3/"

    #Rank refers to which shard number to read.  Rank_index is the index
    rank_index = 0
    
    #index for which rows of the current file to read
    file_pointer = -batch_size

    num_rows = 145

    labels_metadata = np.load("/home/mudigonda/Data/tiramisu_clipped_combined_v2/image_metadata.npy")

    #Contains all labels for all 92,109 images
    labels = np.load("/home/mudigonda/Data/tiramisu_clipped_combined_v2/masks.npy")

    curr_file = np.load(dataset_path+str(training_ranks[rank_index]) + "images.npy")
    curr_image_metadata = np.load(dataset_path+str(training_ranks[rank_index]) + "image_metadata.npy")
    
    reread_npy = False

    while 1:
        file_pointer += batch_size
        if file_pointer + batch_size >= num_rows:
            rank_index += 1
            file_pointer = 0
            reread_npy = True
        if rank_index >= len(training_ranks):
            rank_index=0
            file_pointer=0
            reread_npy = True
        if reread_npy:
            curr_file = np.load(dataset_path+str(training_ranks[rank_index]) + "images.npy")
            curr_image_metadata = np.load(dataset_path+str(training_ranks[rank_index]) + "image_metadata.npy")
            reread_npy = False
        
        #Calculate the current batch's indices "labels"    
        label_indices = np.where((labels_metadata==curr_image_metadata[file_pointer:file_pointer+batch_size,:][:,None]).all(-1))[1]
        #file_pointer+= batch_size
        yield curr_file[file_pointer:file_pointer+batch_size,3:-3,:, [1,3,8]], labels[label_indices].reshape(batch_size,image_height*image_width,1)
        
# -------------------------------------- TRAIN AND TEST TIRAMISU --------------------------------------#

import IPython; IPython.embed()

labels = np.load("/home/mudigonda/Data/tiramisu_clipped_combined_v2/masks.npy")

#Image metadata contains year, month, day, time_step, and lat/ lon data for each crop.  
#See README in $SCRATCH/segmentation_labels/dump_v4 on CORI
image_metadata = np.load("/home/mudigonda/Data/tiramisu_clipped_combined_v2/image_metadata.npy")
labels = labels[:,3:-3,:]

#PERMUTATION OF DATA
#Because the data is too big to be stored and loaded in 1 npy file, it is stored in shards. Each shard has a number associated with it.
#An example file would be 32images.npy and 32image_metadata.npy (for the corresponding metadata).  There are 629 shards.
np.random.seed(12345)
num_shards = 629
shuffle_indices = np.random.permutation(num_shards) 
np.save("./shuffle_indices.npy", shuffle_indices)
ranks = np.arange(628)[shuffle_indices]
train_ranks =ranks[:int(0.8*len(ranks))]
valid_ranks = ranks[int(0.8*len(ranks)):int(0.9*len(ranks))]
test_ranks = ranks[int(0.9*len(ranks)):]


input_shape = (96,144,3)

img_input = Input(shape=input_shape)
blocks=[3,3,4,7,10]
x = create_tiramisu(3, img_input,nb_layers_per_block=blocks, p=0.2, wd=1e-4)
model = Model(img_input, x)

#This is an attempt to weight foreground classes more than background classes... I had to do some hacky stuff to get it to run for segmentation (set the mode to temporal).
#Haven't test this rigorously. Use with caution.

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
cp_callback = keras.callbacks.ModelCheckpoint("/home/mudigonda/Data/multi_channel_training_round1/weights-{epoch:02d}-{val_loss:.2f}-" + str(len(blocks)) +"-" + str(time.time())[-7:-3] + "-.hdf5", monitor='val_loss', verbose=2, save_best_only=False, save_weights_only=False, mode='auto', period=1)

#Create an early stopping callbakc
early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        min_delta=0.0001,
                                                        patience=20,
                                                        verbose=0, mode='auto')

model.compile(loss='sparse_categorical_crossentropy', 
               optimizer=keras.optimizers.RMSprop(1e-3), metrics=["accuracy"])

#This method trains the model assuming that all the training data is loaded in at once (and doesn't use a generator)
#model.fit(trn, trn_labels.squeeze().reshape(trn.shape[0],image_height*image_width,1), 
#    nb_epoch=200,shuffle=True, verbose=2, validation_data=(valid, valid_labels.squeeze().reshape(valid.shape[0],image_height*image_width,1)),
#    callbacks=[cp_callback, early_stopping_callback])

#This method trains the model by loading each batch's data from the generator
model.fit_generator(multi_channel_generator(36, train_ranks),samples_per_epoch=73000, nb_epoch=200, verbose=2, callbacks=[cp_callback, early_stopping_callback], validation_data = multi_channel_generator(36, valid_ranks), nb_val_samples=9200)

