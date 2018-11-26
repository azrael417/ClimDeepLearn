from libs.datasets.coco import read
import tensorflow as tf

file = '~/Tools/FastMaskRCNN/data/coco/records/coco_train2014_00000-of-00033.tfrecord'

with tf.Session() as sess:
     data = read(file)
