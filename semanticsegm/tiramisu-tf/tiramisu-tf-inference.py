# suppress warnings from earlier versions of h5py (imported by tensorflow)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# don't write bytecode for imported modules to disk - different ranks can
#  collide on the file and/or make the file system angry
import sys
sys.dont_write_bytecode = True

import tensorflow as tf
import tensorflow.contrib.keras as tfk
from tensorflow.python.ops import array_ops
import numpy as np
import argparse

# instead of scipy, use PIL directly to save images
try:
    import PIL
    def imsave(filename, data):
        PIL.Image.fromarray(data.astype(np.uint8)).save(filename)
    have_imsave = True
except ImportError:
    have_imsave = False

import h5py as h5
import os
import time

# limit tensorflow spewage to just warnings and errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

#import helpers
try:
    script_path = os.path.dirname(sys.argv[0])
except:
    script_path = '.'
sys.path.append(os.path.join(script_path, '..', 'utils'))
from tiramisu_model import *
from common_helpers import *
from data_helpers import *

#GLOBAL CONSTANTS
image_height_orig = 768
image_width_orig = 1152


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)

#main function
def main(input_path_test, downsampling_fact, downsampling_mode, channels, data_format, label_id, blocks, weights, image_dir, checkpoint_dir, output_graph_file, tst_sz, loss_type, fs_type, batch, batchnorm, dtype, filter_sz, growth, scale_factor):

    #init horovod
    comm_rank = 0
    comm_local_rank = 0
    comm_size = 1
    comm_local_size = 1

    #session config
    sess_config=tf.ConfigProto(inter_op_parallelism_threads=6, #1
                               intra_op_parallelism_threads=1, #6
                               log_device_placement=False,
                               allow_soft_placement=True)
    sess_config.gpu_options.visible_device_list = str(comm_local_rank)
    sess_config.gpu_options.force_gpu_compatible = True

    #get data
    test_graph = tf.Graph()
    tst_data  = load_data(input_path_test, False, tst_sz, False)
    print("Shape of tst_data is {}".format(tst_data.shape[0]))
    print("done.")

    #print some stats
    print("Num workers: {}".format(comm_size))
    print("Local batch size: {}".format(batch))
    if dtype == tf.float32:
        print("Precision: {}".format("FP32"))
    else:
        print("Precision: {}".format("FP16"))
    print("Batch normalization: {}".format(batchnorm))
    print("Blocks: {}".format(blocks))
    print("Growth rate: {}".format(growth))
    print("Filter size: {}".format(filter_sz))
    print("Channels: {}".format(channels))
    print("Loss type: {}".format(loss_type))
    print("Loss weights: {}".format(weights))
    print("Loss scale factor: {}".format(scale_factor))
    print("Num test samples: {}".format(tst_data.shape[0]))

    #compute epochs and stuff:
    if fs_type == "local":
        num_samples = tst_data.shape[0] // comm_local_size
    else:
        num_samples = tst_data.shape[0] // comm_size

    #downsampling? recompute image dims
    image_height =  image_height_orig // downsampling_fact
    image_width = image_width_orig // downsampling_fact
        
    with test_graph.as_default():
        #create readers
        tst_reader = h5_input_reader(input_path_test, channels, weights, dtype, normalization_file="stats.h5", update_on_read=False, label_id=label_id)
        #create datasets
        if fs_type == "local":
            tst_dataset = create_dataset(tst_reader, tst_data, batch, 1, comm_local_size, comm_local_rank, dtype, shuffle=False)
        else:
            tst_dataset = create_dataset(tst_reader, tst_data, batch, 1, comm_size, comm_rank, dtype, shuffle=False)

        #create iterators
        handle = tf.placeholder(tf.string, shape=[], name="iterator-placeholder")
        iterator = tf.data.Iterator.from_string_handle(handle, (dtype, tf.int32, dtype, tf.string),
                                                       ((batch, len(channels), image_height_orig, image_width_orig),
                                                        (batch, image_height_orig, image_width_orig),
                                                        (batch, image_height_orig, image_width_orig),
                                                        (batch))
                                                       )
        next_elem = iterator.get_next()
        
        #if downsampling, do some preprocessing
        if downsampling_fact != 1:
            if downsampling_mode == "scale":
                #do downsampling
                rand_select = tf.cast(tf.one_hot(tf.random_uniform((batch, image_height, image_width), minval=0, maxval=downsampling_fact*downsampling_fact, dtype=tf.int32), depth=downsampling_fact*downsampling_fact, axis=-1), dtype=tf.int32)
                next_elem = (tf.layers.average_pooling2d(next_elem[0], downsampling_fact, downsampling_fact, 'valid', data_format), \
                             tf.reduce_max(tf.multiply(tf.image.extract_image_patches(tf.expand_dims(next_elem[1], axis=-1), \
                                                                                 [1, downsampling_fact, downsampling_fact, 1], \
                                                                                 [1, downsampling_fact, downsampling_fact, 1], \
                                                                                 [1,1,1,1], 'VALID'), rand_select), axis=-1), \
                             tf.squeeze(tf.layers.average_pooling2d(tf.expand_dims(next_elem[2], axis=-1), downsampling_fact, downsampling_fact, 'valid', "channels_last"), axis=-1), \
                             next_elem[3])
            elif downsampling_mode == "center-crop":
                #some parameters
                length = 1./float(downsampling_fact)
                offset = length/2.
                boxes = [[ offset, offset, offset+length, offset+length ]]*batch
                box_ind = list(range(0,batch))
                crop_size = [image_height, image_width]
                
                #be careful with data order
                if data_format=="channels_first":
                    next_elem = (tf.transpose(next_elem[0], perm=[0,2,3,1]), next_elem[1], next_elem[2], next_elem[3])
                    
                #crop
                next_elem = (tf.image.crop_and_resize(next_elem[0], boxes, box_ind, crop_size, method='bilinear', extrapolation_value=0, name="data_cropping"), \
                             ensure_type(tf.squeeze(tf.image.crop_and_resize(tf.expand_dims(next_elem[1],axis=-1), boxes, box_ind, crop_size, method='nearest', extrapolation_value=0, name="label_cropping"), axis=-1), tf.int32), \
                             tf.squeeze(tf.image.crop_and_resize(tf.expand_dims(next_elem[2],axis=-1), boxes, box_ind, crop_size, method='bilinear', extrapolation_value=0, name="weight_cropping"), axis=-1), \
                             next_elem[3])
                
                #be careful with data order
                if data_format=="channels_first":
                    next_elem = (tf.transpose(next_elem[0], perm=[0,3,1,2]), next_elem[1], next_elem[2], next_elem[3])
                    
            else:
                raise ValueError("Error, downsampling mode {} not supported. Supported are [center-crop, scale]".format(downsampling_mode))
        
        #create init handles
        #trn
        tst_iterator = tst_dataset.make_initializable_iterator()
        tst_handle_string = tst_iterator.string_handle()
        tst_init_op = iterator.make_initializer(tst_dataset)

        #compute the input filter number based on number of channels used
        num_channels = len(channels)
        nb_filter = 64

        #set up model
        logit, prediction = create_tiramisu(3, next_elem[0], image_height, image_width, \
                                            num_channels, loss_weights=weights, \
                                            nb_layers_per_block=blocks, p=0.2, wd=1e-4, \
                                            dtype=dtype, batchnorm=batchnorm, growth_rate=growth, \
                                            nb_filter=nb_filter, filter_sz=filter_sz, median_filter=False, data_format=data_format)
        #prediction_argmax = median_pool(prediction_argmax, 3, strides=[1,1,1,1])

        #set up loss
        loss = None
        if loss_type == "weighted":
            #cast weights to FP32
            w_cast = ensure_type(next_elem[2], tf.float32)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=next_elem[1],
                                                          logits=logit,
                                                          weights=w_cast,
                                                          reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
            if scale_factor != 1.0:
                loss *= scale_factor
        elif loss_type == "focal":
            labels_one_hot = tf.contrib.layers.one_hot_encoding(next_elem[1], 3)
            labels_one_hot = ensure_type(labels_one_hot, dtype)
            loss = focal_loss(onehot_labels=labels_one_hot, logits=logit, alpha=1., gamma=2.)
        else:
            raise ValueError("Error, loss type {} not supported.",format(loss_type))

        #set up streaming metrics
        iou_op, iou_update_op = tf.metrics.mean_iou(labels=next_elem[1],
                                                    predictions=tf.argmax(prediction, axis=3),
                                                    num_classes=3,
                                                    weights=None,
                                                    metrics_collections=None,
                                                    updates_collections=None,
                                                    name="iou_score")
        iou_reset_op = tf.variables_initializer([ i for i in tf.local_variables() if i.name.startswith('iou_score/') ])

        #initializers:
        init_op =  tf.global_variables_initializer()
        init_local_op = tf.local_variables_initializer()

        #create image dir if not exists
        if not os.path.isdir(image_dir):
            os.makedirs(image_dir)

         #start session
        with tf.Session(config=sess_config) as sess:
            #initialize
            sess.run([init_op, init_local_op])
            #restore from checkpoint:
            load_model(sess, tf.train.Saver(), checkpoint_dir)
            #create iterator handles
            tst_handle = sess.run(tst_handle_string)
            #init iterators
            sess.run(tst_init_op, feed_dict={handle: tst_handle})

            #remove training nodes
            if output_graph_file:
                print("Storing inference graph to {}.".format(output_graph_file))
                inference_graph_def = tf.graph_util.remove_training_nodes(sess.graph_def, protected_nodes=None)
                #save the inference graph
                with open(output_graph_file, 'wb') as ogf:
                    ogf.write(inference_graph_def.SerializeToString())

            #start inference
            eval_loss = 0.
            eval_steps = 0
            print("Starting evaluation on test set")
            while True:
                try:
                    #construct feed dict
                    _, tmp_loss, tst_model_predictions, tst_model_labels, tst_model_filenames = sess.run([iou_update_op,
                                                                                                          loss,
                                                                                                          prediction,
                                                                                                          next_elem[1],
                                                                                                          next_elem[3]],
                                                                                                          feed_dict={handle: tst_handle})
                    #print some images
                    if have_imsave:
                        imsave(image_dir+'/test_pred_estep'
                               +str(eval_steps)+'_rank'+str(comm_rank)+'.png', np.argmax(tst_model_predictions[0,...],axis=-1)*100)
                        imsave(image_dir+'/test_label_estep'
                               +str(eval_steps)+'_rank'+str(comm_rank)+'.png', tst_model_labels[0,...]*100)
                        imsave(image_dir+'/test_combined_estep'
                               +str(eval_steps)+'_rank'+str(comm_rank)+'.png', plot_colormap[tst_model_labels[0,...],np.argmax(tst_model_predictions[0,...],axis=-1)])
                    else:
                        np.savez(image_dir+'/test_estep'
                                 +str(eval_steps)+'_rank'+str(comm_rank)+'.npz', prediction=np.argmax(tst_model_predictions[...],axis=-1)*100,
                                                                                                 label=tst_model_labels[...]*100, filename=tst_model_filenames)

                    #update loss
                    eval_loss += tmp_loss
                    eval_steps += 1

                except tf.errors.OutOfRangeError:
                    eval_steps = np.max([eval_steps,1])
                    eval_loss /= eval_steps
                    print("COMPLETED: evaluation loss is {}".format(eval_loss))
                    iou_score = sess.run(iou_op)
                    print("COMPLETED: evaluation IoU is {}".format(iou_score))
                    break

if __name__ == '__main__':
    AP = argparse.ArgumentParser()
    AP.add_argument("--blocks",default=[3,3,4,4,7,7],type=int,nargs="*",help="Number of layers per block")
    AP.add_argument("--output",type=str,default='output',help="Defines the location and name of output directory")
    AP.add_argument("--chkpt_dir",type=str,default='checkpoint',help="Defines the location and name of the checkpoint file")
    AP.add_argument("--output_graph",type=str,default=None,help="Filename of the compressed inference graph.")
    AP.add_argument("--test_size",type=int,default=-1,help="How many samples do you want to use for testing?")
    AP.add_argument("--frequencies",default=[0.991,0.0266,0.13],type=float, nargs='*',help="Frequencies per class used for reweighting")
    AP.add_argument("--downsampling",default=1,type=int,help="Downsampling factor for image resolution reduction.")
    AP.add_argument("--downsampling_mode",default="scale",type=str,help="Which mode to use [scale, center-crop].")
    AP.add_argument("--loss",default="weighted",choices=["weighted","focal"],type=str, help="Which loss type to use. Supports weighted, focal [weighted]")
    AP.add_argument("--datadir_test",type=str,help="Path to test data")
    AP.add_argument("--channels",default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],type=int, nargs='*',help="Channels from input images fed to the network. List of numbers between 0 and 15")
    AP.add_argument("--fs",type=str,default="local",help="File system flag: global or local are allowed [local]")
    AP.add_argument("--batch",type=int,default=1,help="Batch size")
    AP.add_argument("--use_batchnorm",action="store_true",help="Set flag to enable batchnorm")
    AP.add_argument("--dtype",type=str,default="float32",choices=["float32","float16"],help="Data type for network")
    AP.add_argument("--filter-sz",type=int,default=3,help="Convolution filter size")
    AP.add_argument("--growth",type=int,default=16,help="Channel growth rate per layer")
    AP.add_argument("--scale_factor",default=0.1,type=float,help="Factor used to scale loss. ")
    AP.add_argument("--data_format", default="channels_first",help="Which data format shall be picked [channels_first, channels_last].")
    AP.add_argument("--label_id", type=int, default=None, help="Allows to select a certain label out of a multi-channel labeled data, \
                    where each channel presents a different label (e.g. for fuzzy labels). \
                    If set to None, the selection will be randomized [None].")
    parsed = AP.parse_args()

    #play with weighting
    weights = [1./x for x in parsed.frequencies]
    weights /= np.sum(weights)

    # convert name of datatype into TF type object
    dtype=getattr(tf, parsed.dtype)

    #invoke main function
    main(input_path_test=parsed.datadir_test,
         downsampling_fact=parsed.downsampling,
         downsampling_mode=parsed.downsampling_mode,
         channels=parsed.channels,
         data_format=parsed.data_format,
         label_id=parsed.label_id,
         blocks=parsed.blocks,
         weights=weights,
         image_dir=parsed.output,
         checkpoint_dir=parsed.chkpt_dir,
         output_graph_file=parsed.output_graph,
         tst_sz=parsed.test_size,
         loss_type=parsed.loss,
         fs_type=parsed.fs,
         batch=parsed.batch,
         batchnorm=parsed.use_batchnorm,
         dtype=dtype,
         filter_sz=parsed.filter_sz,
         growth=parsed.growth,
         scale_factor=parsed.scale_factor)
