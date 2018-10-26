import tensorflow as tf
import tensorflow.contrib.keras as tfk
from tensorflow.contrib.framework import arg_scope
from tensorflow.python.ops import array_ops
from tensorflow.contrib.slim.nets import resnet_v2, vgg
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib import layers as layers_lib

from common_helpers import *

#global parameters
_BATCH_NORM_DECAY = 0.9997
_WEIGHT_DECAY = 5e-4

#atruous spatial pyramid pooling
def atrous_spatial_pyramid_pooling(inputs, output_stride, batch_norm_decay, is_training, depth=256, model_arg_scope=resnet_v2.resnet_arg_scope):
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

        with tf.contrib.slim.arg_scope(model_arg_scope(batch_norm_decay=batch_norm_decay)):
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

    model_arg_scope = resnet_v2.resnet_arg_scope
    if base_architecture == 'resnet_v2_50':
        base_model = resnet_v2.resnet_v2_50
    elif base_architecture == 'resnet_v2_101':
        base_model = resnet_v2.resnet_v2_101
    else:
        raise ValueError("'base_architrecture' {ba} not supported.".format(ba=base_architecture))


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
                inputs = ensure_type(inputs, dtype)
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
        with tf.contrib.slim.arg_scope(model_arg_scope(batch_norm_decay=batch_norm_decay)):
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
        encoder_output = atrous_spatial_pyramid_pooling(net, output_stride, batch_norm_decay, is_training, model_arg_scope=model_arg_scope)

        if data_format == "channels_last":
            decoder_fmt = 'NHWC'
            ch_axis = 3
        else:
            decoder_fmt = 'NCHW'
            ch_axis = 1

        with tf.variable_scope("decoder"):
            with tf.contrib.slim.arg_scope(model_arg_scope(batch_norm_decay=batch_norm_decay)):
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
                            print('ERROR: unknown decoder type:', decoder)
                            assert False
                        sm_logits = tf.nn.softmax(logits, axis=-1)

        return logits, sm_logits

    return model
