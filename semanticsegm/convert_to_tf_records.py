from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import zipfile
import time
import numpy as np
import tensorflow as tf
from six.moves import urllib
from PIL import Image
import skimage.io as io
from matplotlib import pyplot as plt
import glob
import netCDF4 as nc

#from libs.datasets.pycocotools.coco import COCO
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType
from libs.logs.log import LOG

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('vis',  False,
                          'Show some visual masks')

dataset_dir = "~/files_for_first_maskrcnn_test/"

#split name refers to train, val, or test (how the dataset is split)
def _get_dataset_filename(dataset_dir, split_name, shard_id, num_shards):
  output_filename = 'extremes_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, num_shards)
  return os.path.join(dataset_dir, output_filename)


def _get_image_filenames(image_dir):
  return sorted(os.listdir(image_dir))

def load_image(filepath, year, month, day, time_step):
    filepath += "CAM5-1-0.25degree_All-Hist_est1_v3_run2.cam.h2." +"{:04d}-{:02d}-{:02d}-00000.nc".format(year, month, day)
    print(filepath)
    with nc.Dataset(filepath) as fin:
        TMQ = fin['TMQ'][:][time_step:time_step+1,:,:]
    return TMQ


def _int64_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def _process_img_id_string(img_id_string):
  year = int(img_id_string[:4])
  month = int(img_id_string[4:6])
  day = int(img_id_string[6:8])
  time_step = int(img_id_string[8:10])
  return year, month, day, time_step

def _convert_to_tfexample(image_id, image_data,
                           height, width,
                           num_instances, gt_boxes, gt_masks):
  """ just write a raw input"""
  return tf.train.Example(features=tf.train.Features(feature={
    'image/img_id': _int64_feature(image_id),
    'image/encoded': _bytes_feature(tf.compat.as_bytes((image_data*1000).astype('uint8').tostring())),
    'image/height': _int64_feature(height),
    'image/width': _int64_feature(width),
    'label/num_instances': _int64_feature(num_instances),  # N
    'label/gt_boxes': _bytes_feature(tf.compat.as_bytes(gt_boxes.astype('float32').tostring())),  # of shape (N, 5), (x1, y1, x2, y2, classid)
    'label/gt_masks': _bytes_feature(tf.compat.as_bytes(gt_masks.astype('uint8').tostring()))  # of shape (N, height, width)
    #'label/encoded': _bytes_feature(label_data),  # deprecated, this is used for pixel-level segmentation
  }))

dataset_dir = '/home/mudigonda/files_for_first_maskrcnn_test/'
record_dir = dataset_dir + 'records/'
filenames = glob.glob(dataset_dir + '[1-2]*boxes.npy')
num_images = len(filenames)
  
num_images_per_shard = 6
num_shards = int(num_images / float(num_images_per_shard))

print(num_images)
print(num_shards)


for shard_id in range(1,num_shards+1):
  for i in range(num_images_per_shard):
    output_filename = _get_dataset_filename(record_dir, 'train', shard_id, num_shards)
    options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
    with tf.python_io.TFRecordWriter(output_filename, options=options) as tfrecord_writer:
      #Get index of first digit
      data_filename = filenames[(shard_id-1) * num_images_per_shard + i]
      id_start_index = min([i for i, c in enumerate(data_filename) if c.isdigit()])
      img_id_string = data_filename[id_start_index:id_start_index+10]
      img_id = int(img_id_string)

      year, month, day, time_step = _process_img_id_string(img_id_string)

      img = load_image(dataset_dir, year, month, day, time_step)

      height = 768
      width = 1152

      gt_boxes = np.load("/home/mudigonda/files_for_first_maskrcnn_test/{:10d}_instance_boxes.npy".format(img_id)).astype('float32')
      gt_masks = np.load("/home/mudigonda/files_for_first_maskrcnn_test/{:10d}_instance_masks.npy".format(img_id)).astype('float32')
    

      example = _convert_to_tfexample(img_id, img, height, width, 
        gt_boxes.shape[0], gt_boxes, gt_masks)
      tfrecord_writer.write(example.SerializeToString())

print('\nFinished converting the coco dataset!')
