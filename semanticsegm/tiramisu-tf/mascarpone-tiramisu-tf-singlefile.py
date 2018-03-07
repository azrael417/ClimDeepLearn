import tensorflow as tf
import tensorflow.contrib.keras as tfk
from tensorflow.python.ops import array_ops
import numpy as np
import argparse
use_scipy=True
try:
    from scipy.misc import imsave
except:
    use_scipy=False
    
import h5py as h5
import os
import time

#horovod, yes or no?
horovod=True
try:
    import horovod.tensorflow as hvd
except:
    horovod = False

#GLOBAL CONSTANTS
image_height =  768 
image_width = 1152


def focal_loss(onehot_labels, logits, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -sum_i alpha_i * y_i * (1-p_i)^gamma * log(p_i)
                 ,which alpha = 0.25, gamma = 2, p = predictions, y = target_tensor.
    Args:
     onehot_labels: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     logits: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the prediction logits for each class
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    #subtract the mean before softmaxing
    pred = tf.nn.softmax(logits, axis=3)
    #taking the log with some care
    log_pred = tf.log(tf.clip_by_value(pred,1e-8,1.))
    #compute weighted labels:
    weighted_onehot_labels = tf.multiply(onehot_labels,(1-pred)**gamma)
    #compute the product of logs, weights and reweights
    prod = -1. * tf.multiply(tf.multiply(weighted_onehot_labels, log_pred), alpha)
    
    return tf.reduce_mean(tf.reduce_sum(prod,axis=3))


def get_optimizer(opt_type, loss, global_step, learning_rate, momentum=0., LARC_mode="clip", LARC_eta=None, LARC_epsilon=1.):
    #make sure code works for running w/o LARC:
    start_lr = 1.0 if LARC_eta is not None and isinstance(LARC_eta, float) else learning_rate

    #set up optimizers
    if opt_type == "Adam":
        optim = tf.train.RMSPropOptimizer(learning_rate=start_lr)
    elif opt_type == "RMSProp":
        optim = tf.train.RMSPropOptimizer(learning_rate=start_lr)
    elif opt_type == "SGD":
        optim = tf.train.MomentumOptimizer(learning_rate=start_lr, momentum=momentum)
    else:
        raise ValueError("Error, optimizer {} unsupported.".format(opt_type))
        
    #horovod wrapper
    if horovod:
        optim = hvd.DistributedOptimizer(optim)
        
    # LARC gradient re-scaling
    if LARC_eta is not None and isinstance(LARC_eta, float):
        #compute gradients
        grads_and_vars = optim.compute_gradients(loss)
        for idx, (g, v) in enumerate(grads_and_vars):
            if g is not None:
                v_norm = tf.norm(tensor=v, ord=2)
                g_norm = tf.norm(tensor=g, ord=2)
                larc_local_lr = tf.cond(
                    pred = tf.logical_and( tf.not_equal(v_norm, tf.constant(0.0)), tf.not_equal(g_norm, tf.constant(0.0)) ),
                    true_fn = lambda: LARC_eta * v_norm / g_norm,
                    false_fn = lambda: LARC_epsilon)
                if LARC_mode=="scale":
                    effective_lr = larc_local_lr*learning_rate
                else:
                    effective_lr = tf.minimum(larc_local_lr, learning_rate)
                #multiply gradients
                grads_and_vars[idx] = (tf.scalar_mul(effective_lr, g), v)

        #apply gradients:
        train_op = optim.apply_gradients(grads_and_vars, global_step=global_step)
    else:
        #just call minimizer here
        train_op = optim.minimize(loss, global_step=global_step)
    
    #return optimizer
    return train_op


def conv(x, nf, sz, wd, stride=1): 
    return tf.layers.conv2d(inputs=x, filters=nf, kernel_size=sz, strides=(stride,stride),
                            padding='same', data_format='channels_first',
                            kernel_initializer= tfk.initializers.he_uniform(),
                            bias_initializer=tf.initializers.zeros(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=wd)
                            )


def dense_block(n, x, growth_rate, p, wd, training):

    added = []
    for i in range(n):
        with tf.name_scope("bn_relu_conv%i"%i) as scope:
            b = tf.layers.batch_normalization(x, axis=1, training=training)
            b = tf.nn.relu(b)
            b = conv(b, growth_rate, sz=3, wd=wd)
            if p: b = tf.layers.dropout(b, rate=p, training=training)

            x = tf.concat([x, b], axis=1) #was axis=-1. Is that correct?
            added.append(b)

    return x, added


def transition_dn(x, p, wd, training):
    with tf.name_scope("bn_relu_conv") as scope:
        b = tf.layers.batch_normalization(x, axis=1, training=training)
        b = tf.nn.relu(b)
        b = conv(b, x.get_shape().as_list()[1], sz=1, wd=wd, stride=2) #was [-1]. Filters are at 1 now.
        if p: b = tf.layers.dropout(b, rate=p, training=training)
    return b


def down_path(x, nb_layers, growth_rate, p, wd, training):

    skips = []
    for i,n in enumerate(nb_layers):
        with tf.name_scope("DB%i"%i):
            x, added = dense_block(n, x, growth_rate, p, wd, training=training)
            skips.append(x)
        with tf.name_scope("TD%i"%i):
            x = transition_dn(x, p=p, wd=wd, training=training)

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
    
	
def up_path(added,skips,nb_layers,growth_rate,p,wd,training):
    for i,n in enumerate(nb_layers):
        x = transition_up(added,wd,training)
        x = tf.concat([x,skips[i]],axis=1) #was axis=-1. Is that correct?
        x, added = dense_block(n,x,growth_rate,p,wd,training=training)
    return x


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
         growth_rate=16, nb_filter=48, nb_layers_per_block=5, p=None, wd=0., training=True, dtype=tf.float16):
    
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)
    else: nb_layers = [nb_layers_per_block] * nb_dense_block

    # Probably better to do this in the reader
    if dtype != tf.float32:
        img_input = tf.cast(img_input, dtype)

    with tf.variable_scope("tiramisu", custom_getter=float32_variable_storage_getter):

        with tf.variable_scope("conv_input") as scope:
            x = conv(img_input, nb_filter, sz=3, wd=wd)
            if p: x = tf.layers.dropout(x, rate=p, training=training)

        with tf.name_scope("down_path") as scope:
            skips,added = down_path(x, nb_layers, growth_rate, p, wd, training=training)
        
        with tf.name_scope("up_path") as scope:
            x = up_path(added, reverse(skips[:-1]),reverse(nb_layers[:-1]), growth_rate, p, wd,training=training)

        with tf.name_scope("conv_output") as scope:
            x = conv(x,nb_classes,sz=1,wd=wd)
            if p: x = tf.layers.dropout(x, rate=p, training=training)
            _,f,r,c = x.get_shape().as_list()
        #x = tf.reshape(x,[-1,nb_classes,image_height,image_width]) #nb_classes was last before
        x = tf.transpose(x,[0,2,3,1]) #necessary because sparse softmax cross entropy does softmax over last axis

        if x.dtype != tf.float32:
            x = tf.cast(x, tf.float32)
        
    return x, tf.nn.softmax(x)


#Load Data
def load_data(input_path, comm_size, comm_rank, max_files):
    
    #look for labels and data files
    files = sorted([x for x in os.listdir(input_path) if x.startswith("data")])
    
    #we will choose to load only the first p files
    files = files[:max_files]
        
    #convert to numpy
    files = np.asarray(files)

    #PERMUTATION OF DATA
    np.random.seed(12345)
    shuffle_indices = np.random.permutation(len(files))
    np.save("./shuffle_indices.npy", shuffle_indices)
    files = files[shuffle_indices]
    
    #Create train/validation/test split
    size = len(files)
    trn_data = files[:int(0.8 * size)]
    tst_data = files[int(0.8 * size):int(0.9 * size)]
    val_data = files[int(0.9 * size):]
    
    return trn_data, val_data, tst_data


class h5_input_reader(object):
    
    def __init__(self, path, channels, weights, update_on_read=False):
        self.path = path
        self.channels = channels
        self.minvals = np.asarray([np.inf]*len(channels), dtype=np.float32)
        self.maxvals = np.asarray([-np.inf]*len(channels), dtype=np.float32)
        self.update_on_read = update_on_read
        self.weights = weights
    
    def read(self, datafile):
        
        #data
        #begin_time = time.time()
        with h5.File(self.path+'/'+datafile, "r", driver="core", backing_store=False, libver="latest") as f:
            #get min and max values and update stored values
            if self.update_on_read:
                self.minvals = np.minimum(self.minvals, f['climate']['stats'][self.channels,0])
                self.maxvals = np.maximum(self.maxvals, f['climate']['stats'][self.channels,1])
            #get data
            data = f['climate']['data'][self.channels,:,:].astype(np.float32)
            #do min/max normalization
            for c in range(len(self.channels)):
                data[c,:,:] = (data[c,:,:]-self.minvals[c])/(self.maxvals[c]-self.minvals[c])
            
            #get label
            label = f['climate']['labels'][...].astype(np.int32)

            #get weights
            weights = np.zeros(label.shape, dtype=np.float32)
            for idx,w in enumerate(self.weights):
                weights[np.where(label==idx)]=w

        #time
        #end_time = time.time()
        #print "Time to read image %.3f s" % (end_time-begin_time)

        return data, label, weights


def create_dataset(h5ir, datafilelist, batchsize, num_epochs, comm_size, comm_rank, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices(datafilelist)
    if comm_size>1:
        dataset = dataset.shard(comm_size, comm_rank)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.map(lambda dataname: tuple(tf.py_func(h5ir.read, [dataname], [tf.float32, tf.int32, tf.float32])))
    dataset = dataset.batch(batchsize)
    dataset = dataset.repeat(num_epochs)
    
    return dataset


#main function
def main(input_path,blocks,weights,image_dir,checkpoint_dir,trn_sz,learning_rate,loss_type):
    #init horovod
    comm_rank = 0 
    comm_local_rank = 0
    comm_size = 1
    if horovod:
        hvd.init()
        comm_rank = hvd.rank() 
        comm_local_rank = hvd.local_rank()
        comm_size = hvd.size()
        if comm_rank == 0:
            print("Using distributed computation with Horovod: {} total ranks".format(comm_size,comm_rank))
        
    #parameters
    batch = 1
    channels = [0,1,2,10]
    num_epochs = 150
    dtype = tf.float32
    
    #session config
    sess_config=tf.ConfigProto(inter_op_parallelism_threads=2, #1
                               intra_op_parallelism_threads=33, #6
                               log_device_placement=False,
                               allow_soft_placement=True)
    sess_config.gpu_options.visible_device_list = str(comm_local_rank)
    
    #get data
    training_graph = tf.Graph()
    if comm_rank == 0:
        print("Loading data...")
    trn_data, val_data, tst_data = load_data(input_path,comm_size,comm_rank,trn_sz)
    if comm_rank == 0:
        print("Shape of trn_data is {}".format(trn_data.shape[0]))
        print("done.")
    
    #print some stats
    if comm_rank==0:
        print("Learning Rate: {}".format(learning_rate))
        print("Num workers: {}".format(comm_size))
        print("Local batch size: {}".format(batch))
        if dtype == tf.float32:
            print("Precision: {}".format("FP32"))
        else:
            print("Precision: {}".format("FP16"))
        print("Channels: {}".format(channels))
        print("Loss type: {}".format(loss_type))
        print("Loss weights: {}".format(weights))
        print("Num training samples: {}".format(trn_data.shape[0]))
        print("Num validation samples: {}".format(val_data.shape[0]))

    with training_graph.as_default():
        #create datasets
        #files = tf.placeholder(tf.string, shape=[None])
        trn_reader = h5_input_reader(input_path, channels, weights, update_on_read=True)
        trn_dataset = create_dataset(trn_reader, trn_data, batch, num_epochs, comm_size, comm_rank, shuffle=True)
        val_reader = h5_input_reader(input_path, channels, weights, update_on_read=False)
        val_dataset = create_dataset(val_reader, val_data, batch, 1, comm_size, comm_rank, shuffle=False)
        
        #create iterators
        handle = tf.placeholder(tf.string, shape=[], name="iterator-placeholder")
        iterator = tf.data.Iterator.from_string_handle(handle, (tf.float32, tf.int32, tf.float32), 
                                                       ((batch, len(channels), image_height, image_width),
                                                        (batch, image_height, image_width),
                                                        (batch, image_height, image_width))
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

        #set up model
        logit, prediction = create_tiramisu(3, next_elem[0], image_height, image_width, len(channels), loss_weights=weights, nb_layers_per_block=blocks, p=0.2, wd=1e-4, dtype=dtype)
        
        #set up loss
        labels_one_hot = tf.contrib.layers.one_hot_encoding(next_elem[1], 3)
        loss = None
        if loss_type == "weighted":
            loss = tf.losses.softmax_cross_entropy(onehot_labels=labels_one_hot,logits=logit,weights=next_elem[2])
        elif loss_type == "focal":
            loss = focal_loss(onehot_labels=labels_one_hot, logits=logit, alpha=1., gamma=2.)
        else:
            raise ValueError("Error, loss type {} not supported.",format(loss_type))

        #stuff for debugging loss
        #prediction_am = tf.argmax(prediction, axis=3)
        #prediction_onehot =  tf.contrib.layers.one_hot_encoding(prediction_am, 3)
        #prediction_hist = tf.reduce_mean(prediction_onehot, axis=[0,1,2])
        #labels_hist = tf.reduce_mean(labels_one_hot, axis=[0,1,2])

        #set up global step
        global_step = tf.train.get_or_create_global_step()
        
        #set up optimizer
        train_op = get_optimizer("Adam", loss, global_step, learning_rate, LARC_mode="clip", LARC_eta=0.002, LARC_epsilon=1.)
        #set up streaming metrics
        iou_op, iou_update_op = tf.metrics.mean_iou(prediction,labels_one_hot,3,weights=None,metrics_collections=None,updates_collections=None,name="iou_score")
        
        #compute epochs and stuff:
        num_samples = trn_data.shape[0] // comm_size
        num_steps_per_epoch = num_samples // batch
        num_steps = num_epochs*num_steps_per_epoch
        
        #hooks
        #these hooks are essential. regularize the step hook by adding one additional step at the end
        hooks = [tf.train.StopAtStepHook(last_step=num_steps+1)]
        if horovod:
            hooks.append(hvd.BroadcastGlobalVariablesHook(0))
        #initializers:
        init_op =  tf.global_variables_initializer()
        init_local_op = tf.local_variables_initializer()
        
        #checkpointing
        if comm_rank == 0:
            checkpoint_save_freq = num_steps_per_epoch * 2
            checkpoint_saver = tf.train.Saver(max_to_keep = 1000)
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
        

        #start session
        with tf.train.MonitoredTrainingSession(config=sess_config, hooks=hooks) as sess:
            #initialize
            sess.run([init_op, init_local_op])
            #create iterator handles
            trn_handle, val_handle = sess.run([trn_handle_string, val_handle_string])
            #init iterators
            sess.run(trn_init_op, feed_dict={handle: trn_handle})
            sess.run(val_init_op, feed_dict={handle: val_handle})

            #do the training
            epoch = 1
            train_loss = 0.
            start_time = time.time()
            training_start_time = start_time
            while not sess.should_stop():
                
                #training loop
                try:
                    #construct feed dict
                    _, _, train_steps, tmp_loss = sess.run([train_op, iou_update_op, global_step, loss], feed_dict={handle: trn_handle})
                    train_steps_in_epoch = train_steps%num_steps_per_epoch
                    train_loss += tmp_loss
                    
                    if train_steps_in_epoch > 0:
                        #print step report
                        print("REPORT: rank {}, training loss for step {} (of {}) is {}, time {}".format(comm_rank, train_steps, num_steps, train_loss/train_steps_in_epoch,time.time()-training_start_time))
                    else:
                        end_time = time.time()
                        #print epoch report
                        train_loss /= num_steps_per_epoch
                        print("COMPLETED: rank {}, training loss for epoch {} (of {}) is {}, epoch duration {} s".format(comm_rank, epoch, num_epochs, train_loss, end_time - start_time))
                        iou_score = sess.run(iou_op)
                        print("COMPLETED: rank {}, training IoU for epoch {} (of {}) is {}, epoch duration {} s".format(comm_rank, epoch, num_epochs, iou_score, end_time - start_time))
                        start_time = time.time()
                        
                        #evaluation loop
                        eval_loss = 0.
                        eval_steps = 0
                        #update the input reader
                        val_reader.minvals = trn_reader.minvals
                        val_reader.maxvals = trn_reader.maxvals
                        while True:
                            try:
                                #construct feed dict
                                _, tmp_loss, val_model_predictions, val_model_labels = sess.run([iou_update_op, loss, prediction, next_elem[1]], feed_dict={handle: val_handle})
                                if use_scipy:
                                    imsave(image_dir+'/test_pred_epoch'+str(epoch)+'_estep'
                                            +str(eval_steps)+'_rank'+str(comm_rank)+'.png',np.argmax(val_model_predictions[0,...],axis=2)*100)
                                    imsave(image_dir+'/test_label_epoch'+str(epoch)+'_estep'
                                            +str(eval_steps)+'_rank'+str(comm_rank)+'.png',val_model_labels[0,...]*100)
                                else:
                                    np.save(image_dir+'/test_pred_epoch'+str(epoch)+'_estep'
                                            +str(eval_steps)+'_rank'+str(comm_rank)+'.npy',np.argmax(val_model_predictions[0,...],axis=2)*100)
                                    np.save(image_dir+'/test_label_epoch'+str(epoch)+'_estep'
                                            +str(eval_steps)+'_rank'+str(comm_rank)+'.npy',val_model_labels[0,...]*100)
                                eval_loss += tmp_loss
                                eval_steps += 1
                            except tf.errors.OutOfRangeError:
                                eval_steps = np.max([eval_steps,1])
                                eval_loss /= eval_steps
                                print("COMPLETED: rank {}, evaluation loss for epoch {} (of {}) is {}".format(comm_rank, epoch-1, num_epochs, eval_loss))
                                iou_score = sess.run(iou_op)
                                print("COMPLETED: rank {}, evaluation IoU for epoch {} (of {}) is {}".format(comm_rank, epoch-1, num_epochs, iou_score))
                                sess.run(val_init_op, feed_dict={handle: val_handle})
                                break
                                
                        #reset counters
                        epoch += 1
                        train_loss = 0.
                    
                except tf.errors.OutOfRangeError:
                    break

        #test only on rank 0
        #if hvd.rank() == 0:
        #    with tf.Session(config=sess_config) as sess:
        #        #init eval
        #        eval_steps = 0
        #        eval_loss = 0.
        #        #init iterator and variables
        #        sess.run([init_op, init_local_op])
        #        tst_handle = sess.run(tst_handle_string)
        #        sess.run(tst_init_op, feed_dict={handle: tst_handle, tst_feat_placeholder: tst, tst_lab_placeholder: tst_labels})
        #        
        #        #start evaluation
        #        while True:
        #            try:
        #                #construct feed dict
        #                _, tmp_loss = sess.run([iou_update_op, loss], feed_dict={handle: tst_handle})
        #                test_loss += tmp_loss
        #                test_steps += 1
        #            except tf.errors.OutOfRangeError:
        #                test_loss /= test_steps
        #                print("FINAL: test loss for {} epochs is {}".format(epoch-1, test_loss))
        #                iou_score = sess.run([iou_op])
        #                print("FINAL: test IoU for {} epochs is {}".format(epoch-1, iou_score))

if __name__ == '__main__':
    AP = argparse.ArgumentParser()
    AP.add_argument("--lr",default=1e-4,type=float,help="Learning rate")
    AP.add_argument("--blocks",default=[3,3,4,4,7,7,10],type=int,nargs="*",help="Number of layers per block")
    AP.add_argument("--output",type=str,default='output',help="Defines the location and name of output directory")
    AP.add_argument("--chkpt",type=str,default='checkpoint',help="Defines the location and name of the checkpoint directory")
    AP.add_argument("--trn_sz",type=int,default=-1,help="How many samples do you want to use for training? A small number can be used to help debug/overfit")
    AP.add_argument("--frequencies",default=[0.982,0.00071,0.017],type=float, nargs='*',help="Frequencies per class used for reweighting")
    AP.add_argument("--loss",default="weighted",type=str, help="Which loss type to use. Supports weighted, focal [weighted]")
    AP.add_argument("--datadir",type=str,help="Path to input data")
    parsed = AP.parse_args()

    #play with weighting
    weights = [1./x for x in parsed.frequencies]
    weights /= np.sum(weights)
    
    #invoke main function
    main(input_path=parsed.datadir,blocks=parsed.blocks,weights=weights,image_dir=parsed.output,checkpoint_dir=parsed.chkpt,trn_sz=parsed.trn_sz,learning_rate=parsed.lr, loss_type=parsed.loss)
