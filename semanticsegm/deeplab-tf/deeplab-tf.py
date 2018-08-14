# suppress warnings from earlier versions of h5py (imported by tensorflow)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# don't write bytecode for imported modules to disk - different ranks can
#  collide on the file and/or make the file system angry
import sys
sys.dont_write_bytecode = True

import tensorflow as tf
import tensorflow.contrib.keras as tfk
from tensorflow.contrib.framework import arg_scope
from tensorflow.python.ops import array_ops
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib import layers as layers_lib
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

use_nvtx = False
if (use_nvtx):
  import cupy.cuda.nvtx as nvtx
else:
  class nvtx_dummy:
    def RangePush(self, name, color):
      pass
    def RangePop(self):
      pass
  nvtx = nvtx_dummy()

#horovod, yes or no?
horovod=True
try:
    import horovod.tensorflow as hvd
except:
    horovod = False

#import helpers
try:
    script_path = os.path.dirname(sys.argv[0])
except:
    script_path = '.'
sys.path.append(os.path.join(script_path, '..', 'utils'))
from climseg_helpers import *
import graph_flops

#GLOBAL CONSTANTS
image_height =  768 
image_width = 1152

#arch specific
_BATCH_NORM_DECAY = 0.9997
_WEIGHT_DECAY = 5e-4


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


def ensure_type(input, dtype):
    if input.dtype != dtype:
        return tf.cast(input, dtype)
    else:
        return input


def atrous_spatial_pyramid_pooling(inputs, output_stride, batch_norm_decay, is_training, depth=256):
    """Atrous Spatial Pyramid Pooling.
    Args:
      inputs: A tensor of size [batch, height, width, channels].
      output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
      the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
      batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
      is_training: A boolean denoting whether the input is for training.
      depth: The depth of the ResNet unit output.
    Returns:
      The atrous spatial pyramid pooling output.
    """
    with tf.variable_scope("aspp"):
        if output_stride not in [8, 16]:
            raise ValueError('output_stride must be either 8 or 16.')
        
        atrous_rates = [6, 12, 18]
        if output_stride == 8:
            atrous_rates = [2*rate for rate in atrous_rates]

        with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
            with arg_scope([layers.batch_norm], is_training=is_training):
                inputs_size = tf.shape(inputs)[1:3]
                # (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18) when output stride = 16.
                # the rates are doubled when output stride = 8.
                conv_1x1 = layers_lib.conv2d(inputs, depth, [1, 1], stride=1, scope="conv_1x1")
                conv_3x3_1 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[0], scope='conv_3x3_1')
                conv_3x3_2 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[1], scope='conv_3x3_2')
                conv_3x3_3 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[2], scope='conv_3x3_3')

                # (b) the image-level features
                with tf.variable_scope("image_level_features"):
                    # global average pooling
                    image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keepdims=True)
                    # 1x1 convolution with 256 filters( and batch normalization)
                    image_level_features = layers_lib.conv2d(image_level_features, depth, [1, 1], stride=1, scope='conv_1x1')
                    # bilinearly upsample features
                    image_level_features = ensure_type(image_level_features, tf.float32)
                    image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')
                    image_level_features = ensure_type(image_level_features, inputs.dtype)

                net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
                net = layers_lib.conv2d(net, depth, [1, 1], stride=1, scope='conv_1x1_concat')

    return net


def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      initializer=initializer, regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
    return variable


#generate deeplab model
def deeplab_v3_plus_generator(num_classes,
                              output_stride,
                              base_architecture,
                              decoder,
                              batchnorm,
                              pre_trained_model,
                              batch_norm_decay,
                              data_format='channels_last'):
    """Generator for DeepLab v3 plus models.
    Args:
      num_classes: The number of possible classes for image classification.
      output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
        the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
      base_architecture: The architecture of base Resnet building block.
      pre_trained_model: The path to the directory that contains pre-trained models.
      batch_norm_decay: The moving average decay when estimating layer activation
        statistics in batch normalization.
      data_format: The input format ('channels_last', 'channels_first', or None).
        If set to None, the format is dependent on whether a GPU is available.
        Only 'channels_last' is supported currently.
    Returns:
      The model function that takes in `inputs` and `is_training` and
      returns the output tensor of the DeepLab v3 model.
    """
    if data_format is None:
        # data_format = (
        #     'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
        pass

    if batch_norm_decay is None:
        batch_norm_decay = _BATCH_NORM_DECAY

    if base_architecture not in ['resnet_v2_50', 'resnet_v2_101']:
        raise ValueError("'base_architrecture' must be either 'resnet_v2_50' or 'resnet_v2_50'.")

    if base_architecture == 'resnet_v2_50':
        base_model = resnet_v2.resnet_v2_50
    else:
        base_model = resnet_v2.resnet_v2_101

    base_architecture = 'getter_scope/' + base_architecture

    def model(inputs, is_training, dtype=tf.float32):
        # we can't directly control the instantiation of batchnorms, but
        #  we can monkey-patch the TF module to turn them into nop's
        if not batchnorm:
            from tensorflow.contrib.framework.python.ops import add_arg_scope
            from tensorflow.contrib.layers.python.layers import layers as layers_to_hack
            @add_arg_scope
            def dummy_batch_norm(input, *args, **kwargs):
                # ideally we'd just pass the input straight through, but
                #  activations in deep networks can overflow fp16's range
                input = tf.multiply(input, 0.5)
                return input
            orig_batch_norm = layers_to_hack.batch_norm
            layers_to_hack.batch_norm = dummy_batch_norm
        with tf.variable_scope('getter_scope', custom_getter=float32_variable_storage_getter):
            if dtype != tf.float32:
                inputs = tf.cast(inputs, dtype)
            m = model_fp32(inputs, is_training)
        if not batchnorm:
            layers_to_hack.batch_norm = orig_batch_norm
        return m

    def model_fp32(inputs, is_training):
        """Constructs the ResNet model given the inputs."""
        if data_format == 'channels_first':
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
            # This provides a large performance boost on GPU. See
            # https://www.tensorflow.org/performance/performance_guide#data_formats
            inputs = tf.transpose(inputs, [0, 2, 3, 1])

        # tf.logging.info('net shape: {}'.format(inputs.shape))
        # encoder
        with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
            logits, end_points = base_model(inputs,
                                            num_classes=None,
                                            is_training=is_training,
                                            global_pool=False,
                                            output_stride=output_stride)

        if is_training:
            if pre_trained_model:
                exclude = [base_architecture + '/logits', 'global_step']
                variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
                tf.train.init_from_checkpoint(pre_trained_model, {v.name.split(':')[0]: v for v in variables_to_restore})

        inputs_size = tf.shape(inputs)[1:3]
        net = end_points[base_architecture + '/block4']
        encoder_output = atrous_spatial_pyramid_pooling(net, output_stride, batch_norm_decay, is_training)

        #decoder_fmt = 'NHWC'
        #ch_axis = 3
        decoder_fmt = 'NCHW'
        ch_axis = 1

        with tf.variable_scope("decoder"):
            with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
                with arg_scope([layers.batch_norm], is_training=is_training):
                    with tf.variable_scope("low_level_features"):
                        low_level_features = end_points[base_architecture + '/block1/unit_3/bottleneck_v2/conv1']
                        low_level_features_size = tf.shape(low_level_features)[1:3]
                        if decoder_fmt == 'NCHW':
                            low_level_features = tf.transpose(low_level_features, [ 0, 3, 1, 2 ])
                        low_level_features = layers_lib.conv2d(low_level_features, 48,
                                                               [1, 1], stride=1, scope='conv_1x1',
                                                               data_format=decoder_fmt)

                    with tf.variable_scope("upsampling_logits"):
                        if decoder == 'bilinear':
                            assert decoder_fmt == 'NHWC'
                            encoder_output = ensure_type(encoder_output, tf.float32)
                            net = tf.image.resize_bilinear(encoder_output, low_level_features_size, name='upsample_1')
                            net = ensure_type(net, low_level_features.dtype)
                            net = tf.concat([net, low_level_features], axis=3, name='concat')
                            net = layers_lib.conv2d(net, 256, [3, 3], stride=1, scope='conv_3x3_1')
                            net = layers_lib.conv2d(net, 256, [3, 3], stride=1, scope='conv_3x3_2')
                            net = layers_lib.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv_1x1')
                            net = ensure_type(net, tf.float32)
                            logits = tf.image.resize_bilinear(net, inputs_size, name='upsample_2')
                            logits = ensure_type(logits, low_level_features.dtype)
                        elif decoder.startswith('deconv'):
                            if decoder_fmt == 'NCHW':
                                encoder_output = tf.transpose(encoder_output, [ 0, 3, 1, 2 ])
                                inputs = tf.transpose(inputs, [ 0, 3, 1, 2 ])
                            # expect encoder output at 1/8x input, low level
                            #  features as 1/4x input
                            #print 'SIZES', encoder_output.shape.as_list(), low_level_features.shape.as_list(), inputs.shape.as_list()
                            assert 8*encoder_output.shape.as_list()[2] == inputs.shape.as_list()[2]
                            assert 4*low_level_features.shape.as_list()[2] == inputs.shape.as_list()[2]
                            encoder_channels = encoder_output.shape.as_list()[ch_axis]
                            low_level_channels = low_level_features.shape.as_list()[ch_axis]
                            inputs_channels = inputs.shape.as_list()[ch_axis]
                            net = tf.layers.conv2d_transpose(inputs=encoder_output,
                                                             strides=(2,2),
                                                             kernel_size=(3,3),
				                             padding='same',
                                                             data_format='channels_last' if (decoder_fmt == 'NHWC') else 'channels_first',
                                                             filters=encoder_channels,
				                             kernel_initializer=tfk.initializers.he_uniform(),
				                             bias_initializer=tf.initializers.zeros(),
                                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=_WEIGHT_DECAY))
                            net = tf.concat([net, low_level_features], axis=ch_axis, name='concat')
                            net = layers_lib.conv2d(net, 256, [3, 3], stride=1, scope='conv_3x3_1',
                                                    data_format=decoder_fmt)
                            net = layers_lib.conv2d(net, 256, [3, 3], stride=1, scope='conv_3x3_2',
                                                    data_format=decoder_fmt)
                            # two 2x deconvs instead of the 4x bilinear scale
                            net = tf.layers.conv2d_transpose(inputs=net,
                                                             strides=(2,2),
                                                             kernel_size=(3,3),
				                             padding='same',
                                                             data_format='channels_last' if (decoder_fmt == 'NHWC') else 'channels_first',
                                                             filters=256,
				                             kernel_initializer=tfk.initializers.he_uniform(),
				                             bias_initializer=tf.initializers.zeros(),
                                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=_WEIGHT_DECAY))
                            net = tf.layers.conv2d_transpose(inputs=net,
                                                             strides=(2,2),
                                                             kernel_size=(3,3),
				                             padding='same',
                                                             data_format='channels_last' if (decoder_fmt == 'NHWC') else 'channels_first',
                                                             filters=256,
				                             kernel_initializer=tfk.initializers.he_uniform(),
				                             bias_initializer=tf.initializers.zeros(),
                                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=_WEIGHT_DECAY))
                            if decoder == 'deconv1x':
                                # incorporate input data at this level
                                skip = tf.layers.conv2d(inputs, 64, [3, 3],
                                                        padding='same',
                                                        data_format='channels_last' if (decoder_fmt == 'NHWC') else 'channels_first',
				                        kernel_initializer=tfk.initializers.he_uniform(),
				                        bias_initializer=tf.initializers.zeros(),
                                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=_WEIGHT_DECAY))
                                skip = tf.layers.conv2d(skip, 128, [3, 3],
                                                        padding='same',
                                                        data_format='channels_last' if (decoder_fmt == 'NHWC') else 'channels_first',
				                        kernel_initializer=tfk.initializers.he_uniform(),
				                        bias_initializer=tf.initializers.zeros(),
                                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=_WEIGHT_DECAY))
                                net = tf.concat([net, skip], axis=ch_axis)
                                net = tf.layers.conv2d(net, 256, [3, 3],
                                                       padding='same',
                                                       data_format='channels_last' if (decoder_fmt == 'NHWC') else 'channels_first',
				                       kernel_initializer=tfk.initializers.he_uniform(),
				                       bias_initializer=tf.initializers.zeros(),
                                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=_WEIGHT_DECAY))
                                net = tf.layers.conv2d(net, 256, [3, 3],
                                                       padding='same',
                                                       data_format='channels_last' if (decoder_fmt == 'NHWC') else 'channels_first',
				                       kernel_initializer=tfk.initializers.he_uniform(),
				                       bias_initializer=tf.initializers.zeros(),
                                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=_WEIGHT_DECAY))
                            
                            logits = layers_lib.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv_1x1',
                                                       data_format=decoder_fmt)
                            if decoder_fmt == 'NCHW':
                                logits = tf.transpose(logits, [ 0, 2, 3, 1 ])
                        else:
                            print 'ERROR: unknown decoder type:', decoder
                            assert False
                        sm_logits = tf.nn.softmax(logits)

        return logits, sm_logits

    return model


def create_dataset(h5ir, datafilelist, batchsize, num_epochs, comm_size, comm_rank, dtype, shuffle=False):
    if comm_size > 1:
        # use an equal number of files per shard, leaving out any leftovers
        per_shard = len(datafilelist) // comm_size
        sublist = datafilelist[0:per_shard * comm_size]
        dataset = tf.data.Dataset.from_tensor_slices(sublist)
        dataset = dataset.shard(comm_size, comm_rank)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(datafilelist)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.map(map_func=lambda dataname: tuple(tf.py_func(h5ir.read, [dataname], [dtype, tf.int32, dtype, tf.string])),
                          num_parallel_calls = 4)
    dataset = dataset.prefetch(16)
    # make sure all batches are equal in size
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batchsize))
    dataset = dataset.repeat(num_epochs)
    
    return dataset


#                                       label predict color
colormap = np.array([[[  0,  0,  0],  #   0      0     black
                      [255,  0,255],  #   0      1     purple
                      [  0,255,255]], #   0      2     cyan
                     [[  0,255,  0],  #   1      0     green
                      [128,128,128],  #   1      1     grey
                      [255,255,  0]], #   1      2     yellow
                     [[255,  0,  0],  #   2      0     red
                      [  0,  0,255],  #   2      1     blue
                      [255,255,255]], #   2      2     white
                     ])

#main function
def main(input_path_train, input_path_validation, channels, weights, image_dir, checkpoint_dir, trn_sz, val_sz, loss_type, cluster_loss_weight, model, decoder, fs_type, optimizer, batch, batchnorm, num_epochs, dtype, chkpt, disable_checkpoints, disable_imsave, tracing, trace_dir, output_sampling, scale_factor):
    #init horovod
    nvtx.RangePush("init horovod", 1)
    comm_rank = 0 
    comm_local_rank = 0
    comm_size = 1
    comm_local_size = 1
    if horovod:
        hvd.init()
        comm_rank = hvd.rank() 
        comm_local_rank = hvd.local_rank()
        comm_size = hvd.size()
        #not all horovod versions have that implemented
        try:
            comm_local_size = hvd.local_size()
        except:
            comm_local_size = 1
        if comm_rank == 0:
            print("Using distributed computation with Horovod: {} total ranks".format(comm_size,comm_rank))
    nvtx.RangePop() # init horovod
        
    #parameters
    per_rank_output = False
    loss_print_interval = 10
    
    #session config
    sess_config=tf.ConfigProto(inter_op_parallelism_threads=6, #1
                               intra_op_parallelism_threads=1, #6
                               log_device_placement=False,
                               allow_soft_placement=True)
    sess_config.gpu_options.visible_device_list = str(comm_local_rank)
    sess_config.gpu_options.force_gpu_compatible = True

    #get data
    training_graph = tf.Graph()
    if comm_rank == 0:
        print("Loading data...")
    trn_data = load_data(input_path_train, True, trn_sz, horovod)
    val_data = load_data(input_path_validation, False, val_sz, horovod)
    if comm_rank == 0:    
        print("Shape of trn_data is {}".format(trn_data.shape[0]))
        print("Shape of val_data is {}".format(val_data.shape[0]))
        print("done.")

    #print some stats
    if comm_rank==0:
        print("Num workers: {}".format(comm_size))
        print("Local batch size: {}".format(batch))
        if dtype == tf.float32:
            print("Precision: {}".format("FP32"))
        else:
            print("Precision: {}".format("FP16"))
        print("Decoder: {}".format(decoder))
        print("Batch normalization: {}".format(batchnorm))
        print("Channels: {}".format(channels))
        print("Loss type: {}".format(loss_type))
        print("Loss weights: {}".format(weights))
        print("Loss scale factor: {}".format(scale_factor))
        print("Cluster loss weight: {}".format(cluster_loss_weight))
        print("Output sampling target: {}".format(output_sampling))
        #print optimizer parameters
        for k,v in optimizer.iteritems():
            print("Solver Parameters: {k}: {v}".format(k=k,v=v))
        print("Num training samples: {}".format(trn_data.shape[0]))
        print("Num validation samples: {}".format(val_data.shape[0]))
        print("Disable checkpoints: {}".format(disable_checkpoints))
        print("Disable image save: {}".format(disable_imsave))

    #compute epochs and stuff:
    if fs_type == "local":
        num_samples = trn_data.shape[0] // comm_local_size
    else:
        num_samples = trn_data.shape[0] // comm_size
    num_steps_per_epoch = num_samples // batch
    num_steps = num_epochs*num_steps_per_epoch
    if per_rank_output:
        print("Rank {} does {} steps per epoch".format(comm_rank,
                                                       num_steps_per_epoch))

    with training_graph.as_default():
        nvtx.RangePush("TF Init", 3)
        #create readers
        trn_reader = h5_input_reader(input_path_train, channels, weights, dtype, normalization_file="stats.h5", update_on_read=False, sample_target=output_sampling)
        val_reader = h5_input_reader(input_path_validation, channels, weights, dtype, normalization_file="stats.h5", update_on_read=False)
        #create datasets
        if fs_type == "local":
            trn_dataset = create_dataset(trn_reader, trn_data, batch, num_epochs, comm_local_size, comm_local_rank, dtype, shuffle=True)
            val_dataset = create_dataset(val_reader, val_data, batch, 1, comm_local_size, comm_local_rank, dtype, shuffle=False)
        else:
            trn_dataset = create_dataset(trn_reader, trn_data, batch, num_epochs, comm_size, comm_rank, dtype, shuffle=True)
            val_dataset = create_dataset(val_reader, val_data, batch, 1, comm_size, comm_rank, dtype, shuffle=False)
        
        #create iterators
        handle = tf.placeholder(tf.string, shape=[], name="iterator-placeholder")
        iterator = tf.data.Iterator.from_string_handle(handle, (dtype, tf.int32, dtype, tf.string),
                                                       ((batch, len(channels), image_height, image_width),
                                                        (batch, image_height, image_width),
                                                        (batch, image_height, image_width),
                                                        (batch))
                                                       )
        next_elem = iterator.get_next()
        
        #create init handles
        #trn
        trn_iterator = trn_dataset.make_initializable_iterator()
        trn_handle_string = trn_iterator.string_handle()
        trn_init_op = iterator.make_initializer(trn_dataset)
        #val
        val_iterator = val_dataset.make_initializable_iterator()
        val_handle_string = val_iterator.string_handle()
        val_init_op = iterator.make_initializer(val_dataset)

        #compute the input filter number based on number of channels used
        num_channels = len(channels)

        #set up model
        model = deeplab_v3_plus_generator(num_classes=3, output_stride=8, 
                                          base_architecture=model,
                                          decoder=decoder,
                                          batchnorm=batchnorm,
                                          pre_trained_model=None, 
                                          batch_norm_decay=None, data_format='channels_first')

        logit, prediction = model(next_elem[0], True, dtype)

        #logit, prediction = create_tiramisu(3, next_elem[0], image_height, image_width, num_channels, loss_weights=weights, nb_layers_per_block=blocks, p=0.2, wd=1e-4, dtype=dtype, batchnorm=batchnorm, growth_rate=growth, nb_filter=nb_filter, filter_sz=filter_sz)
        
        #set up loss
        loss = None
        
        #cast the logits to fp32
        logit = tf.cast(logit, tf.float32)

        if loss_type == "weighted":
            #cast weights to FP32
            w_cast = tf.cast(next_elem[2], tf.float32)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=next_elem[1], 
                                                          logits=logit, 
                                                          weights=w_cast, 
                                                          reduction=tf.losses.Reduction.SUM)
            if scale_factor != 1.0:
                loss *= scale_factor
            #unweighted = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=next_elem[1],
            #                                                            logits=logit)
            #w_cast = tf.cast(next_elem[2], tf.float32)
            #weighted = tf.multiply(unweighted, w_cast)
            #if output_sampling:
            #    loss = tf.reduce_sum(weighted)
            #else:
            #    # TODO: do we really need to normalize this?
            #    #scale_factor = 1. / weighted.shape.num_elements()
            #    loss = tf.reduce_sum(weighted) * scale_factor
            #tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
        elif loss_type == "focal":
            #one-hot-encode
            labels_one_hot = tf.contrib.layers.one_hot_encoding(next_elem[1], 3)
            #cast to FP32
            labels_one_hot = tf.cast(labels_one_hot, tf.float32)
            loss = focal_loss(onehot_labels=labels_one_hot, logits=logit, alpha=1., gamma=2.)
        else:
            raise ValueError("Error, loss type {} not supported.",format(loss_type))

        #if cluster loss is enabled
        if cluster_loss_weight > 0.0:
            loss = loss + cluster_loss_weight * cluster_loss(prediction, 5, padding="SAME", data_format="NHWC", name="cluster_loss")


        flops = graph_flops.graph_flops(format='NHWC',
                                        batch=batch,
                                        sess_config=sess_config)

        flops *= comm_size
        if comm_rank == 0:
            print 'training flops: {:.3f} TF/step'.format(flops * 1e-12)

        if horovod:
            loss_avg = hvd.allreduce(tf.cast(loss, tf.float32))
        else:
            loss_avg = tf.identity(loss)

        #set up global step - keep on CPU
        with tf.device('/device:CPU:0'):
            global_step = tf.train.get_or_create_global_step()

        #set up optimizer
        if optimizer['opt_type'].startswith("LARC"):
            if comm_rank==0:
                print("Enabling LARC")
            train_op, lr = get_larc_optimizer(optimizer, loss, global_step,
                                              num_steps_per_epoch, horovod)
        else:
            train_op, lr = get_optimizer(optimizer, loss, global_step,
                                         num_steps_per_epoch, horovod)

        #set up streaming metrics
        iou_op, iou_update_op = tf.metrics.mean_iou(labels=next_elem[1],
                                                    predictions=tf.argmax(prediction, axis=3),
                                                    num_classes=3,
                                                    weights=None,
                                                    metrics_collections=None,
                                                    updates_collections=None,
                                                    name="iou_score")
        iou_reset_op = tf.variables_initializer([ i for i in tf.local_variables() if i.name.startswith('iou_score/') ])

        if horovod:
            iou_avg = hvd.allreduce(iou_op)
        else:
            iou_avg = tf.identity(iou_op)

        with tf.device('/device:GPU:0'):
            mem_usage_ops = [ tf.contrib.memory_stats.MaxBytesInUse(),
                              tf.contrib.memory_stats.BytesLimit() ]

        #hooks
        #these hooks are essential. regularize the step hook by adding one additional step at the end
        hooks = [tf.train.StopAtStepHook(last_step=num_steps+1)]
        #bcast init for bcasting the model after start
        if horovod:
            init_bcast = hvd.broadcast_global_variables(0)
        #initializers:
        init_op =  tf.global_variables_initializer()
        init_local_op = tf.local_variables_initializer()
        
        #checkpointing
        if comm_rank == 0:
            checkpoint_save_freq = 5*num_steps_per_epoch
            checkpoint_saver = tf.train.Saver(max_to_keep = 1000)
            if (not disable_checkpoints):
                hooks.append(tf.train.CheckpointSaverHook(checkpoint_dir=checkpoint_dir, save_steps=checkpoint_save_freq, saver=checkpoint_saver))
            #create image dir if not exists
            if not os.path.isdir(image_dir):
                os.makedirs(image_dir)
        
        ##DEBUG
        ##summary
        #if comm_rank == 0:
        #    print("write graph for debugging")
        #    tf.summary.scalar("loss",loss)
        #    summary_op = tf.summary.merge_all()
        #    #hooks.append(tf.train.SummarySaverHook(save_steps=num_steps_per_epoch, summary_writer=summary_writer, summary_op=summary_op))
        #    with tf.Session(config=sess_config) as sess:
        #        sess.run([init_op, init_local_op])
        #        #create iterator handles
        #        trn_handle = sess.run(trn_handle_string)
        #        #init iterators
        #        sess.run(trn_init_op, feed_dict={handle: trn_handle, datafiles: trn_data, labelfiles: trn_labels})
        #        #summary:
        #        sess.run(summary_op, feed_dict={handle: trn_handle})
        #        #summary file writer
        #        summary_writer = tf.summary.FileWriter('./logs', sess.graph)
        ##DEBUG

        if tracing is not None:
            import tracehook
            tracing_hook = tracehook.TraceHook(steps_to_trace=tracing,
                                               cache_traces=True,
                                               trace_dir=trace_dir)
            hooks.append(tracing_hook)

        # instead of averaging losses over an entire epoch, use a moving
        #  window average
        recent_losses = []
        loss_window_size = 10

        #start session
        with tf.train.MonitoredTrainingSession(config=sess_config, hooks=hooks) as sess:
            #initialize
            sess.run([init_op, init_local_op])
            #restore from checkpoint:
            if comm_rank == 0 and not disable_checkpoints:
                load_model(sess, checkpoint_saver, checkpoint_dir)
            #broadcast loaded model variables
            if horovod:
                sess.run(init_bcast)
            #create iterator handles
            trn_handle, val_handle = sess.run([trn_handle_string, val_handle_string])
            #init iterators
            sess.run(trn_init_op, feed_dict={handle: trn_handle})
            sess.run(val_init_op, feed_dict={handle: val_handle})

            nvtx.RangePop() # TF Init

            # figure out what step we're on (it won't be 0 if we are
            #  restoring from a checkpoint) so we can count from there
            train_steps = sess.run([global_step])[0]

            #do the training
            epoch = 1
            step = 1

            prev_mem_usage = 0
            t_sustained_start = time.time()
            r_peak = 0

            nvtx.RangePush("Training Loop", 4)
            nvtx.RangePush("Epoch", epoch)
            start_time = time.time()
            while not sess.should_stop():
                
                #training loop
                try:
                    nvtx.RangePush("Step", step)
                    #construct feed dict
                    t_inst_start = time.time()
                    _, tmp_loss, cur_lr = sess.run([train_op,
                                                    (loss if per_rank_output else loss_avg),
                                                    lr],
                                                   feed_dict={handle: trn_handle})
                    t_inst_end = time.time()
                    mem_used = sess.run(mem_usage_ops)
                    train_steps += 1
                    train_steps_in_epoch = train_steps%num_steps_per_epoch
                    recent_losses = [ tmp_loss ] + recent_losses[0:loss_window_size-1]
                    train_loss = sum(recent_losses) / len(recent_losses)
                    nvtx.RangePop() # Step
                    step += 1

                    r_inst = 1e-12 * flops / (t_inst_end-t_inst_start)
                    r_peak = max(r_peak, r_inst)
                    
                    #print step report
                    eff_steps = train_steps_in_epoch if (train_steps_in_epoch > 0) else num_steps_per_epoch
                    if (train_steps % loss_print_interval) == 0:
                        mem_used = sess.run(mem_usage_ops)
                        if per_rank_output:
                            print("REPORT: rank {}, training loss for step {} (of {}) is {}, time {:.3f}".format(comm_rank, train_steps, num_steps, train_loss, time.time()-start_time))
                        else:
                            if comm_rank == 0:
                                if mem_used[0] > prev_mem_usage:
                                    print("memory usage: {:.2f} GB / {:.2f} GB".format(mem_used[0] / 2.0**30, mem_used[1] / 2.0**30))
                                    prev_mem_usage = mem_used[0]
                                print("REPORT: training loss for step {} (of {}) is {}, time {:.3f}, r_inst {:.3f}, r_peak {:.3f}, lr {:.2g}".format(train_steps, num_steps, train_loss, time.time()-start_time, r_inst, r_peak, cur_lr))

                    #do the validation phase
                    if train_steps_in_epoch == 0:
                        end_time = time.time()
                        #print epoch report
                        if per_rank_output:
                            print("COMPLETED: rank {}, training loss for epoch {} (of {}) is {}, time {:.3f}, r_sust {:.3f}".format(comm_rank, epoch, num_epochs, train_loss, time.time() - start_time, 1e-12 * flops * num_steps_per_epoch / (end_time-t_sustained_start)))
                        else:
                            if comm_rank == 0:
                                print("COMPLETED: training loss for epoch {} (of {}) is {}, time {:.3f}, r_sust {:.3f}".format(epoch, num_epochs, train_loss, time.time() - start_time, 1e-12 * flops * num_steps_per_epoch / (end_time-t_sustained_start)))
                        
                        #evaluation loop
                        eval_loss = 0.
                        eval_steps = 0
                        nvtx.RangePush("Eval Loop", 7)
                        while True:
                            try:
                                #construct feed dict
                                _, tmp_loss, val_model_predictions, val_model_labels, val_model_filenames = sess.run([iou_update_op,
                                                                                                                      (loss if per_rank_output else loss_avg),
                                                                                                                      prediction,
                                                                                                                      next_elem[1],
                                                                                                                      next_elem[3]],
                                                                                                                      feed_dict={handle: val_handle})
                                
                                #print some images
                                if comm_rank == 0 and not disable_imsave:
                                    if have_imsave:
                                        imsave(image_dir+'/test_pred_epoch'+str(epoch)+'_estep'
                                               +str(eval_steps)+'_rank'+str(comm_rank)+'.png',np.argmax(val_model_predictions[0,...],axis=2)*100)
                                        imsave(image_dir+'/test_label_epoch'+str(epoch)+'_estep'
                                               +str(eval_steps)+'_rank'+str(comm_rank)+'.png',val_model_labels[0,...]*100)
                                        imsave(image_dir+'/test_combined_epoch'+str(epoch)+'_estep'
                                               +str(eval_steps)+'_rank'+str(comm_rank)+'.png',colormap[val_model_labels[0,...],np.argmax(val_model_predictions[0,...],axis=2)])
                                    else:
                                        np.savez(image_dir+'/test_epoch'+str(epoch)+'_estep'
                                                 +str(eval_steps)+'_rank'+str(comm_rank)+'.npz', prediction=np.argmax(val_model_predictions[0,...],axis=2)*100, 
                                                                                                 label=val_model_labels[0,...]*100,
                                                                                                 filename=val_model_filenames[0])

                                eval_loss += tmp_loss
                                eval_steps += 1
                            except tf.errors.OutOfRangeError:
                                eval_steps = np.max([eval_steps,1])
                                eval_loss /= eval_steps
                                if per_rank_output:
                                    print("COMPLETED: rank {}, evaluation loss for epoch {} (of {}) is {}".format(comm_rank, epoch, num_epochs, eval_loss))
                                else:
                                    if comm_rank == 0:
                                        print("COMPLETED: evaluation loss for epoch {} (of {}) is {}".format(epoch, num_epochs, eval_loss))
                                if per_rank_output:
                                    iou_score = sess.run(iou_op)
                                    print("COMPLETED: rank {}, evaluation IoU for epoch {} (of {}) is {}".format(comm_rank, epoch, num_epochs, iou_score))
                                else:
                                    iou_score = sess.run(iou_avg)
                                    if comm_rank == 0:
                                        print("COMPLETED: evaluation IoU for epoch {} (of {}) is {}".format(epoch, num_epochs, iou_score))
                                sess.run(iou_reset_op)
                                sess.run(val_init_op, feed_dict={handle: val_handle})
                                break
                        nvtx.RangePop() # Eval Loop
                                
                        #reset counters
                        epoch += 1
                        step = 0
                        t_sustained_start = time.time()

                        nvtx.RangePop() # Epoch
                        nvtx.RangePush("Epoch", epoch)
                    
                except tf.errors.OutOfRangeError:
                    break

            nvtx.RangePop() # Epoch
            nvtx.RangePop() # Training Loop

        # write any cached traces to disk
        if tracing is not None:
            tracing_hook.write_traces()

if __name__ == '__main__':
    AP = argparse.ArgumentParser()
    AP.add_argument("--lr",default=1e-4,type=float,help="Learning rate")
    AP.add_argument("--output",type=str,default='output',help="Defines the location and name of output directory")
    AP.add_argument("--chkpt",type=str,default='checkpoint',help="Defines the location and name of the checkpoint file")
    AP.add_argument("--chkpt_dir",type=str,default='checkpoint',help="Defines the location and name of the checkpoint file")
    AP.add_argument("--trn_sz",type=int,default=-1,help="How many samples do you want to use for training? A small number can be used to help debug/overfit")
    AP.add_argument("--val_sz",type=int,default=-1,help="How many samples do you want to use for validation?")
    AP.add_argument("--frequencies",default=[0.991,0.0266,0.13],type=float, nargs='*',help="Frequencies per class used for reweighting")
    AP.add_argument("--loss",default="weighted",choices=["weighted","focal"],type=str, help="Which loss type to use. Supports weighted, focal [weighted]")
    AP.add_argument("--cluster_loss_weight",default=0.0, type=float, help="Weight for cluster loss [0.0]")
    AP.add_argument("--datadir_train",type=str,help="Path to training data")
    AP.add_argument("--datadir_validation",type=str,help="Path to validation data")
    AP.add_argument("--channels",default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],type=int, nargs='*',help="Channels from input images fed to the network. List of numbers between 0 and 15")
    AP.add_argument("--fs",type=str,default="local",help="File system flag: global or local are allowed [local]")
   # AP.add_argument("--optimizer",type=str,default="LARC-Adam",help="Optimizer flag: Adam, RMS, SGD are allowed. Prepend with LARC- to enable LARC [LARC-Adam]")
    AP.add_argument("--optimizer",action=StoreDictKeyPair)
    AP.add_argument("--model",type=str,default="resnet_v2_101",help="Pick base model [resnet_v2_50, resnet_v2_101].")
    AP.add_argument("--decoder",type=str,default="bilinear",help="Pick decoder [bilinear,deconv,deconv1x]")
    AP.add_argument("--epochs",type=int,default=150,help="Number of epochs to train")
    AP.add_argument("--batch",type=int,default=1,help="Batch size")
    AP.add_argument("--use_batchnorm",action="store_true",help="Set flag to enable batchnorm")
    AP.add_argument("--dtype",type=str,default="float32",choices=["float32","float16"],help="Data type for network")
    AP.add_argument("--disable_checkpoints",action='store_true',help="Flag to disable checkpoint saving/loading")
    AP.add_argument("--disable_imsave",action='store_true',help="Flag to disable image saving")
    AP.add_argument("--disable_horovod",action='store_true',help="Flag to disable horovod")
    AP.add_argument("--tracing",type=str,help="Steps or range of steps to trace")
    AP.add_argument("--trace-dir",type=str,help="Directory where trace files should be written")
    AP.add_argument("--gradient-lag",type=int,default=0,help="Steps to lag gradient updates")
    AP.add_argument("--sampling",type=int,help="Target number of pixels from each class to sample")
    AP.add_argument("--scale_factor",default=0.1,type=float,help="Factor used to scale loss. ")
    parsed = AP.parse_args()

    #play with weighting
    weights = [1./x for x in parsed.frequencies]
    weights /= np.sum(weights)

    # convert name of datatype into TF type object
    dtype=getattr(tf, parsed.dtype)

    #check if we want horovod to be disabled
    if parsed.disable_horovod:
        horovod = False

    #invoke main function
    main(input_path_train=parsed.datadir_train,
         input_path_validation=parsed.datadir_validation,
         channels=parsed.channels,
         weights=weights,
         image_dir=parsed.output,
         checkpoint_dir=parsed.chkpt_dir,
         trn_sz=parsed.trn_sz,
         val_sz=parsed.val_sz,
         loss_type=parsed.loss,
         cluster_loss_weight=parsed.cluster_loss_weight,
         model=parsed.model,
         decoder=parsed.decoder,
         fs_type=parsed.fs,
         optimizer=parsed.optimizer,
         num_epochs=parsed.epochs,
         batch=parsed.batch,
         batchnorm=parsed.use_batchnorm,
         dtype=dtype,
         chkpt=parsed.chkpt,
         disable_checkpoints=parsed.disable_checkpoints,
         disable_imsave=parsed.disable_imsave,
         tracing=parsed.tracing,
         trace_dir=parsed.trace_dir,
         output_sampling=parsed.sampling,
         scale_factor=parsed.scale_factor)
