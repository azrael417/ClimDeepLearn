#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import functools
import os, sys
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from time import gmtime, strftime
import netCDF4 as nc
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType


#sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import libs.configs.config_v1 as cfg
import libs.datasets.dataset_factory as datasets
import libs.nets.nets_factory as network 

import libs.preprocessings.coco_v1 as coco_preprocess
import libs.nets.pyramid_network as pyramid_network
import libs.nets.resnet_v1 as resnet_v1
from train.train_utils import _configure_learning_rate, _configure_optimizer, \
  _get_variables_to_train, _get_init_fn, get_var_list_to_restore

from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from libs.datasets import download_and_convert_coco
#from libs.datasets.download_and_convert_coco import _cat_id_to_cls_name
from libs.visualization.pil_utils import cat_id_to_cls_name, draw_img, draw_bbox
import glob

def read(tfrecords_filename):

  if not isinstance(tfrecords_filename, list):
    tfrecords_filename = [tfrecords_filename]
  filename_queue = tf.train.string_input_producer(
    tfrecords_filename, num_epochs=100)

  options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
  reader = tf.TFRecordReader(options=options)
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    features={
      'image/img_id': tf.FixedLenFeature([], tf.int64),
      'image/encoded': tf.FixedLenFeature([], tf.string),
      'image/height': tf.FixedLenFeature([], tf.int64),
      'image/width': tf.FixedLenFeature([], tf.int64),
      'label/num_instances': tf.FixedLenFeature([], tf.int64),
      'label/gt_masks': tf.FixedLenFeature([], tf.string),
      'label/gt_boxes': tf.FixedLenFeature([], tf.string)
      #'label/encoded': tf.FixedLenFeature([], tf.string),
      })
  # image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
  # img_id = tf.cast(features['image/img_id'], tf.int32)
  # ih = tf.cast(features['image/height'], tf.int32)
  # iw = tf.cast(features['image/width'], tf.int32)
  # num_instances = tf.cast(features['label/num_instances'], tf.int32)
  # image = tf.decode_raw(features['image/encoded'], tf.float32)
  # imsize = tf.size(image)
  # image = tf.reshape(image, (ih, iw, 1))
  # #image = tf.cond(tf.equal(imsize, ih * iw), \
  # #        lambda: tf.image.grayscale_to_rgb(tf.reshape(image, (ih, iw, 1))), \
  # #        lambda: tf.reshape(image, (ih, iw, 3)))

  # gt_boxes = tf.decode_raw(features['label/gt_boxes'], tf.float32)
  # gt_boxes = tf.reshape(gt_boxes, [num_instances, 5])
  # gt_masks = tf.decode_raw(features['label/gt_masks'], tf.float32)
  # #gt_masks = tf.cast(gt_masks, tf.int32)
  # gt_masks = tf.reshape(gt_masks, [num_instances, ih, iw])
  
  #return image, ih, iw, gt_boxes, gt_masks, num_instances, img_id
  return features

with tf.Session() as sess:
  tfrecords_filename = '~/Tools/FastMaskRCNN/data/coco/records/coco_train2014_00000-of-00033.tfrecord'

  features = read(tfrecords_filename)

  img_id = tf.cast(features['image/img_id'], tf.int32)
  ih = tf.cast(features['image/height'], tf.int32)
  iw = tf.cast(features['image/width'], tf.int32)
  num_instances = tf.cast(features['label/num_instances'], tf.int32)
  image = tf.decode_raw(features['image/encoded'], tf.uint8)
  imsize = tf.size(image)
  image = tf.cond(tf.equal(imsize, ih * iw), \
          lambda: tf.image.grayscale_to_rgb(tf.reshape(image, (ih, iw, 1))), \
          lambda: tf.reshape(image, (ih, iw, 3)))

  gt_boxes = tf.decode_raw(features['label/gt_boxes'], tf.float32)
  gt_boxes = tf.reshape(gt_boxes, [num_instances, 5])
  gt_masks = tf.decode_raw(features['label/gt_masks'], tf.float32)
  gt_masks = tf.cast(gt_masks, tf.int32)
  gt_masks = tf.reshape(gt_masks, [num_instances, ih, iw])

  
  data_queue = tf.RandomShuffleQueue(capacity=32, min_after_dequeue=16,
            dtypes=(
                image.dtype, ih.dtype, iw.dtype, 
                gt_boxes.dtype, gt_masks.dtype, 
                num_instances.dtype, img_id.dtype)) 
  
  enqueue_op = data_queue.enqueue((image, ih, iw, gt_boxes, gt_masks, num_instances, img_id))
  data_queue_runner = tf.train.QueueRunner(data_queue, [enqueue_op] * 4)
  tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, data_queue_runner)
  (image, ih, iw, gt_boxes, gt_masks, num_instances, img_id) =  data_queue.dequeue()
  im_shape = tf.shape(image)
  image = tf.reshape(image, (im_shape[0], im_shape[1], im_shape[2], 3))

  #image, ih, iw, gt_boxes, gt_masks, num_instances, img_id = tf.train.shuffle_batch([image, ih, iw, gt_boxes, gt_masks, num_instances, img_id], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)

  init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
  sess.run(init_op)
  # Create a coordinator and run all QueueRunner objects
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  for batch_index in range(5):
        image, ih, iw, gt_boxes, gt_masks, num_instances, img_id = sess.run([image, ih, iw, gt_boxes, gt_masks, num_instances, img_id])
	import IPython; IPython.embed()

  # Stop the threads
  coord.request_stop()
  # Wait for threads to stop
  coord.join(threads)
  sess.close()
