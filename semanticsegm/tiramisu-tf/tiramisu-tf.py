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
from climseg_helpers import *
import graph_flops

#GLOBAL CONSTANTS
image_height =  768 
image_width = 1152


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


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
        x = tf.transpose(x,[0,2,3,1]) #necessary because sparse softmax cross entropy does softmax over last axis

        if median_filter:
            x = median_pool(x, 3, [1,1,1,1])

    return x, tf.nn.softmax(x)



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
def main(input_path_train, input_path_validation, channels, blocks, weights, image_dir, checkpoint_dir, trn_sz, loss_type, cluster_loss_weight, fs_type, optimizer, batch, batchnorm, num_epochs, dtype, chkpt, filter_sz, growth, disable_checkpoints, disable_imsave, tracing, trace_dir, output_sampling, scale_factor):

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
    trn_data  = load_data(input_path_train, True, trn_sz)
    val_data  = load_data(input_path_validation, False)
    if comm_rank == 0:    
        print("Shape of trn_data is {}".format(trn_data.shape[0]))
        print("done.")

    #print some stats
    if comm_rank==0:
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
        print("Cluster loss weight: {}".format(cluster_loss_weight))
        print("Output sampling target: {}".format(output_sampling))
        #print optimizer parameters
        for k,v in optimizer.iteritems():
            print("Solver Parameters: {k}: {v}".format(k=k,v=v))
        #print("Optimizer type: {}".format(optimizer['opt_type']))
        print("Num training samples: {}".format(trn_data.shape[0]))
        print("Num validation samples: {}".format(val_data.shape[0]))
        print("Disable checkpoints: {}".format(disable_checkpoints))
        print("Disable image save: {}".format(disable_imsave))

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
        nb_filter = 64

        #set up model
        logit, prediction = create_tiramisu(3, next_elem[0], image_height, image_width, num_channels, loss_weights=weights, nb_layers_per_block=blocks, p=0.2, wd=1e-4, dtype=dtype, batchnorm=batchnorm, growth_rate=growth, nb_filter=nb_filter, filter_sz=filter_sz, median_filter=False)
        prediction_argmax = tf.argmax(prediction, axis=3)
        prediction_argmax = median_pool(prediction_argmax, 3, strides=[1,1,1,1])

        #set up loss
        loss = None
        if loss_type == "weighted":
            logit = tf.cast(logit, tf.float32)
            unweighted = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=next_elem[1],
                                                                        logits=logit)
            w_cast = tf.cast(next_elem[2], tf.float32)
            weighted = tf.multiply(unweighted, w_cast)
            if output_sampling:
                loss = tf.reduce_sum(weighted)
            else:
                # TODO: do we really need to normalize this?
                #scale_factor = 1. / weighted.shape.num_elements()
                loss = tf.reduce_sum(weighted) * scale_factor
            tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
        elif loss_type == "focal":
            labels_one_hot = tf.contrib.layers.one_hot_encoding(next_elem[1], 3)
            labels_one_hot = tf.cast(labels_one_hot, dtype)
            loss = focal_loss(onehot_labels=labels_one_hot, logits=logit, alpha=1., gamma=2.)
        else:
            raise ValueError("Error, loss type {} not supported.",format(loss_type))

        #if cluster loss is enabled
        if cluster_loss_weight > 0.0:
            loss = loss + cluster_loss_weight * cluster_loss(prediction, 5, padding="SAME", data_format="NHWC", name="cluster_loss")

        flops = graph_flops.graph_flops(format='NCHW',
                                        batch=batch)
        if horovod:
            flops *= hvd.size()
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
            train_op = get_larc_optimizer(optimizer, loss, global_step)
        else:
            train_op = get_optimizer(optimizer, loss, global_step)

        #set up streaming metrics
        iou_op, iou_update_op = tf.metrics.mean_iou(labels=next_elem[1],
                                                    predictions=prediction_argmax,
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

        #compute epochs and stuff:
        if fs_type == "local":
            num_samples = trn_data.shape[0] // comm_local_size
        else:
            num_samples = trn_data.shape[0] // comm_size
        num_steps_per_epoch = num_samples // batch
        num_steps = num_epochs*num_steps_per_epoch
        if per_rank_output:
            print("Rank {} does {} steps per epoch".format(comm_rank, num_steps_per_epoch))
        
        #hooks
        #these hooks are essential. regularize the step hook by adding one additional step at the end
        hooks = [tf.train.StopAtStepHook(last_step=num_steps+1)]
        #bcast init for bcasting the model after start
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

            t_sustained_start = time.time()

            nvtx.RangePush("Training Loop", 4)
            nvtx.RangePush("Epoch", epoch)
            start_time = time.time()
            while not sess.should_stop():
                
                #training loop
                try:
                    nvtx.RangePush("Step", step)
                    #construct feed dict
                    t_inst_start = time.time()
                    _, tmp_loss = sess.run([train_op,
                                            (loss if per_rank_output else loss_avg)],
                                           feed_dict={handle: trn_handle})
                    t_inst_end = time.time()
                    train_steps += 1
                    train_steps_in_epoch = train_steps%num_steps_per_epoch
                    recent_losses = [ tmp_loss ] + recent_losses[0:loss_window_size-1]
                    train_loss = sum(recent_losses) / len(recent_losses)
                    nvtx.RangePop() # Step
                    step += 1
                    
                    #print step report
                    eff_steps = train_steps_in_epoch if (train_steps_in_epoch > 0) else num_steps_per_epoch
                    if (train_steps % loss_print_interval) == 0:
                        if per_rank_output:
                            print("REPORT: rank {}, training loss for step {} (of {}) is {}, time {:.3f}".format(comm_rank, train_steps, num_steps, train_loss, time.time()-start_time))
                        else:
                            if comm_rank == 0:
                                print("REPORT: training loss for step {} (of {}) is {}, time {:.3f}, r_inst {:.3f}".format(train_steps, num_steps, train_loss, time.time()-start_time, 1e-12 * flops / (t_inst_end-t_inst_start) ))

                    #do the validation phase
                    if train_steps_in_epoch == 0:
                        end_time = time.time()
                        #print epoch report
                        if per_rank_output:
                            print("COMPLETED: rank {}, training loss for epoch {} (of {}) is {}, time {:.3f}, r_sust {:.3f}".format(comm_rank, epoch, num_epochs, train_loss, time.time() - start_time, 1e-12 * flops * num_steps_per_epoch / (end_time-t_sustained_start) ))
                        else:
                            if comm_rank == 0:
                                print("COMPLETED: training loss for epoch {} (of {}) is {}, time {:.3f}, r_sust {:.3f}".format(epoch, num_epochs, train_loss, time.time() - start_time, time.time() - start_time, 1e-12 * flops * num_steps_per_epoch / (end_time-t_sustained_start) ))
                        
                        #evaluation loop
                        eval_loss = 0.
                        eval_steps = 0
                        nvtx.RangePush("Eval Loop", 7)
                        while True:
                            try:
                                #construct feed dict
                                _, tmp_loss, val_model_predictions, val_model_labels, val_model_filenames = sess.run([iou_update_op,
                                                                                                                      (loss if per_rank_output else loss_avg),
                                                                                                                      prediction_argmax,
                                                                                                                      next_elem[1], 
                                                                                                                      next_elem[3]],
                                                                                                                      feed_dict={handle: val_handle})

                                #print some images
                                if comm_rank == 0 and not disable_imsave:
                                    if have_imsave:
                                        imsave(image_dir+'/test_pred_epoch'+str(epoch)+'_estep'
                                               +str(eval_steps)+'_rank'+str(comm_rank)+'.png',val_model_predictions[0,...]*100)
                                        imsave(image_dir+'/test_label_epoch'+str(epoch)+'_estep'
                                               +str(eval_steps)+'_rank'+str(comm_rank)+'.png',val_model_labels[0,...]*100)
                                        imsave(image_dir+'/test_combined_epoch'+str(epoch)+'_estep'
                                               +str(eval_steps)+'_rank'+str(comm_rank)+'.png',colormap[val_model_labels[0,...],val_model_predictions[0,...]])
                                    else:
                                        np.savez(image_dir+'/test_epoch'+str(epoch)+'_estep'
                                                 +str(eval_steps)+'_rank'+str(comm_rank)+'.npz', prediction=val_model_predictions[0,...]*100, 
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
    AP.add_argument("--blocks",default=[3,3,4,4,7,7,10],type=int,nargs="*",help="Number of layers per block")
    AP.add_argument("--output",type=str,default='output',help="Defines the location and name of output directory")
    AP.add_argument("--chkpt",type=str,default='checkpoint',help="Defines the location and name of the checkpoint file")
    AP.add_argument("--chkpt_dir",type=str,default='checkpoint',help="Defines the location and name of the checkpoint file")
    AP.add_argument("--trn_sz",type=int,default=-1,help="How many samples do you want to use for training? A small number can be used to help debug/overfit")
    AP.add_argument("--frequencies",default=[0.991,0.0266,0.13],type=float, nargs='*',help="Frequencies per class used for reweighting")
    AP.add_argument("--loss",default="weighted",choices=["weighted","focal"],type=str, help="Which loss type to use. Supports weighted, focal [weighted]")
    AP.add_argument("--cluster_loss_weight",default=0.0, type=float, help="Weight for cluster loss [0.0]")
    AP.add_argument("--datadir_train",type=str,help="Path to training data")
    AP.add_argument("--datadir_validation",type=str,help="Path to validation data")
    AP.add_argument("--channels",default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],type=int, nargs='*',help="Channels from input images fed to the network. List of numbers between 0 and 15")
    AP.add_argument("--fs",type=str,default="local",help="File system flag: global or local are allowed [local]")
    #AP.add_argument("--optimizer",type=str,default="LARC-Adam",help="Optimizer flag: Adam, RMS, SGD are allowed. Prepend with LARC- to enable LARC [LARC-Adam]")
    AP.add_argument("--optimizer",action=StoreDictKeyPair)
    AP.add_argument("--epochs",type=int,default=150,help="Number of epochs to train")
    AP.add_argument("--batch",type=int,default=1,help="Batch size")
    AP.add_argument("--use_batchnorm",action="store_true",help="Set flag to enable batchnorm")
    AP.add_argument("--dtype",type=str,default="float32",choices=["float32","float16"],help="Data type for network")
    AP.add_argument("--filter-sz",type=int,default=3,help="Convolution filter size")
    AP.add_argument("--growth",type=int,default=16,help="Channel growth rate per layer")
    AP.add_argument("--disable_checkpoints",action='store_true',help="Flag to disable checkpoint saving/loading")
    AP.add_argument("--disable_imsave",action='store_true',help="Flag to disable image saving")
    AP.add_argument("--tracing",type=str,help="Steps or range of steps to trace")
    AP.add_argument("--trace-dir",type=str,help="Directory where trace files should be written")
    AP.add_argument("--sampling",type=int,help="Target number of pixels from each class to sample")
    AP.add_argument("--scale_factor",default=0.1,type=float,help="Factor used to scale loss. ")
    parsed = AP.parse_args()

    #play with weighting
    weights = [1./x for x in parsed.frequencies]
    weights /= np.sum(weights)

    # convert name of datatype into TF type object
    dtype=getattr(tf, parsed.dtype)

    #invoke main function
    main(input_path_train=parsed.datadir_train,
         input_path_validation=parsed.datadir_validation,
         channels=parsed.channels,
         blocks=parsed.blocks,
         weights=weights,
         image_dir=parsed.output,
         checkpoint_dir=parsed.chkpt_dir,
         trn_sz=parsed.trn_sz,
         loss_type=parsed.loss,
         cluster_loss_weight=parsed.cluster_loss_weight,
         fs_type=parsed.fs,
         optimizer=parsed.optimizer,
         num_epochs=parsed.epochs,
         batch=parsed.batch,
         batchnorm=parsed.use_batchnorm,
         dtype=dtype,
         chkpt=parsed.chkpt,
         filter_sz=parsed.filter_sz,
         growth=parsed.growth,
         disable_checkpoints=parsed.disable_checkpoints,
         disable_imsave=parsed.disable_imsave,
         tracing=parsed.tracing,
         trace_dir=parsed.trace_dir,
         output_sampling=parsed.sampling,
         scale_factor=parsed.scale_factor)
