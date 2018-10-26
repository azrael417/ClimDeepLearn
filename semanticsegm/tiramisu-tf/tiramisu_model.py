import tensorflow as tf
import tensorflow.contrib.keras as tfk

from common_helpers import *

def conv(x, nf, sz, wd, stride=1):
    return tf.layers.conv2d(inputs=x, filters=nf, kernel_size=sz, strides=(stride,stride),
                            padding='same', data_format='channels_first',
                            kernel_initializer= tfk.initializers.he_uniform(),
                            bias_initializer=tf.initializers.zeros(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=wd)
                            )

def dense_block(n, x, growth_rate, p, wd, training, bn=False, filter_sz=3):

    added = []
    for i in range(n):
        if bn:
            with tf.name_scope("conv_bn_relu%i"%i) as scope:
                b = conv(x, growth_rate, sz=filter_sz, wd=wd)
                b = tf.layers.batch_normalization(b, axis=1, training=training)
                b = tf.nn.relu(b)
                if p: b = tf.layers.dropout(b, rate=p, training=training)
        else:
            with tf.name_scope("conv_relu%i"%i) as scope:
                b = conv(x, growth_rate, sz=filter_sz, wd=wd)
                b = tf.nn.relu(b)
                if p: b = tf.layers.dropout(b, rate=p, training=training)

        x = tf.concat([x, b], axis=1) #was axis=-1. Is that correct?
        added.append(b)

    return x, added


def transition_dn(x, p, wd, training, bn=False):
    if bn:
        with tf.name_scope("conv_bn_relu") as scope:
            b = conv(x, x.get_shape().as_list()[1], sz=1, wd=wd, stride=2) #was [-1]. Filters are at 1 now.
            b = tf.layers.batch_normalization(b, axis=1, training=training)
            b = tf.nn.relu(b)
            if p: b = tf.layers.dropout(b, rate=p, training=training)
    else:
        with tf.name_scope("conv_relu") as scope:
            b = conv(x, x.get_shape().as_list()[1], sz=1, wd=wd, stride=2)
            b = tf.nn.relu(b)
            if p: b = tf.layers.dropout(b, rate=p, training=training)
    return b


def down_path(x, nb_layers, growth_rate, p, wd, training, bn=False, filter_sz=3):

    skips = []
    for i,n in enumerate(nb_layers):
        with tf.name_scope("DB%i"%i):
            x, added = dense_block(n, x, growth_rate, p, wd, training=training, bn=bn, filter_sz=filter_sz)
            skips.append(x)
        with tf.name_scope("TD%i"%i):
            x = transition_dn(x, p=p, wd=wd, training=training, bn=bn)

    return skips, added


def reverse(a):
	return list(reversed(a))


def transition_up(added,wd,training):
    x = tf.concat(added,axis=1)
    _, ch, r, c = x.get_shape().as_list()
    x = tf.layers.conv2d_transpose(inputs=x,strides=(2,2),kernel_size=(3,3),
				   padding='same', data_format='channels_first', filters=ch,
				   kernel_initializer=tfk.initializers.he_uniform(),
				   bias_initializer=tf.initializers.zeros(),
                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=wd)
                   )
    return x


def up_path(added,skips,nb_layers,growth_rate,p,wd,training,bn=False,filter_sz=3):
    for i,n in enumerate(nb_layers):
        x = transition_up(added,wd,training)
        x = tf.concat([x,skips[i]],axis=1) #was axis=-1. Is that correct?
        x, added = dense_block(n,x,growth_rate,p,wd,training=training,bn=bn,filter_sz=filter_sz)
    return x


def median_pool(x, filter_size, strides=[1,1,1,1]):
    x_size = x.get_shape().as_list()

    #if 3D input, expand dims first
    if len(x_size) == 3:
        x = tf.expand_dims(x, axis=-1)
    patches = tf.extract_image_patches(x, [1, filter_size, filter_size, 1], strides, 4*[1], 'SAME', name="median_pool")
    #if 4D input, we need to reshape
    if len(x_size) == 4:
        patches = tf.reshape(patches, x_size[0:3]+[filter_size*filter_size]+[x_size[3]])
    #no matter whether 3 or 4D input, always axis 3 has to be pooled over
    medians = tf.contrib.distributions.percentile(patches, 50, axis=3, keep_dims=False)
    return medians


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


def create_tiramisu(nb_classes, img_input, height, width, nc, loss_weights, nb_dense_block=6,
                    growth_rate=16, nb_filter=48, nb_layers_per_block=5, p=None, wd=0., training=True, batchnorm=False, dtype=tf.float16, filter_sz=3, median_filter=False):

    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)
    else: nb_layers = [nb_layers_per_block] * nb_dense_block

    with tf.variable_scope("tiramisu", custom_getter=float32_variable_storage_getter):

        with tf.variable_scope("conv_input") as scope:
            x = conv(img_input, nb_filter, sz=filter_sz, wd=wd)
            if batchnorm:
                x = tf.layers.batch_normalization(x, axis=1, training=training)
            x = tf.nn.relu(x)
            if p: x = tf.layers.dropout(x, rate=p, training=training)

        with tf.name_scope("down_path") as scope:
            skips,added = down_path(x, nb_layers, growth_rate, p, wd, training=training, bn=batchnorm, filter_sz=filter_sz)

        with tf.name_scope("up_path") as scope:
            x = up_path(added, reverse(skips[:-1]),reverse(nb_layers[:-1]), growth_rate, p, wd, training=training, bn=batchnorm, filter_sz=filter_sz)

        with tf.name_scope("conv_output") as scope:
            x = conv(x,nb_classes,sz=1,wd=wd)
            if p: x = tf.layers.dropout(x, rate=p, training=training)
            _,f,r,c = x.get_shape().as_list()
        #x = tf.reshape(x,[-1,nb_classes,image_height,image_width]) #nb_classes was last before
        x = ensure_type(tf.transpose(x,[0,2,3,1]), tf.float32) #necessary because sparse softmax cross entropy does softmax over last axis

        #if median_filter:
        #    x = median_pool(x, 3, [1,1,1,1])

    return x, tf.nn.softmax(x)