import tensorflow as tf
import tensorflow.contrib.keras as tfk
import numpy as np
from scipy.misc import imsave
import h5py as h5
import os

#horovod, yes or no?
horovod=True
try:
    import horovod.tensorflow as hvd
    print("Enabling Horovod Support")
except:
    horovod = False
    print("Disabling Horovod Support")

#GLOBAL CONSTANTS
#image_height = 96
#image_width = 144
image_height =  768 
image_width = 1152
#comm_rank = 0
#comm_local_rank = 0
#comm_size = 1


def conv(x, nf, sz, wd, stride=1): 
    return tf.layers.conv2d(inputs=x, filters=nf, kernel_size=sz, strides=(stride,stride), padding='same',
                            kernel_initializer= tfk.initializers.he_uniform(),
                            bias_initializer=tf.initializers.zeros(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=wd)
                            )

def dense_block(n, x, growth_rate, p, wd, training):

    added = []
    for i in range(n):
        with tf.name_scope("bn_relu_conv%i"%i) as scope:
            b = tf.layers.batch_normalization(x, training=training)
            b = tf.nn.relu(b)
            b = conv(b, growth_rate, sz=3, wd=wd)
            if p: b = tf.layers.dropout(b, rate=p, training=training)

            x = tf.concat([x, b], axis=-1)
            added.append(b)

    return x, added

def transition_dn(x, p, wd, training):
    with tf.name_scope("bn_relu_conv") as scope:
        b = tf.layers.batch_normalization(x, training=training)
        b = tf.nn.relu(b)
        b = conv(b, x.get_shape().as_list()[-1], sz=1, wd=wd, stride=2)
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
    x = tf.concat(added,axis=-1) 
    _, r, c, ch = x.get_shape().as_list()
    x = tf.layers.conv2d_transpose(inputs=x,strides=(2,2),kernel_size=(3,3),
				   padding='same',filters=ch,
				   kernel_initializer=tfk.initializers.he_uniform(),
				   bias_initializer=tf.initializers.zeros(),
                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=wd)
                   )
    return x 
    
	
def up_path(added,skips,nb_layers,growth_rate,p,wd,training):
    for i,n in enumerate(nb_layers):
	x = transition_up(added,wd,training)
	x = tf.concat([x,skips[i]],axis=-1)
	x, added = dense_block(n,x,growth_rate,p,wd,training=training)
    return x

def create_tiramisu(nb_classes, img_input, height, width, nc, nb_dense_block=6, 
         growth_rate=16, nb_filter=48, nb_layers_per_block=5, p=None, wd=0., training=True):
    
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)
    else: nb_layers = [nb_layers_per_block] * nb_dense_block

    with tf.variable_scope("conv_input") as scope:
        #init_w = tfk.initializers.he_uniform()
        #init_b = tf.initializers.zeros()
        #x = tf.nn.conv2d(input=img_input, 
        #                filter=tf.Variable(init_w([3,3,nc,nb_filter]),dtype=tf.float32),
        #                strides=[1, 1, 1, 1],
        #                padding='SAME')
        #x = tf.nn.bias_add(x, tf.Variable(init_b(nb_filter),dtype=tf.float32))
        x = conv(img_input, nb_filter, sz=3, wd=wd)
        if p: x = tf.layers.dropout(x, rate=p, training=training)

    with tf.name_scope("down_path") as scope:
        skips,added = down_path(x, nb_layers, growth_rate, p, wd, training=training)
    
    with tf.name_scope("up_path") as scope:
        x = up_path(added, reverse(skips[:-1]),reverse(nb_layers[:-1]), growth_rate, p, wd,training=training)

    with tf.name_scope("conv_output") as scope:
        x = conv(x,nb_classes,sz=1,wd=wd)
        if p: x = tf.layers.dropout(x, rate=p, training=training)
        _,r,c,f = x.get_shape().as_list()
    x = tf.reshape(x,[-1,image_height,image_width,nb_classes])
    return x, tf.nn.softmax(x)


#Load Data
def load_data():
    #images from directory
    input_path = "/global/cscratch1/sd/amahesh/segm_h5_v3"
    
    #look for labels and data files
    labelfiles = sorted([x for x in os.listdir(input_path) if x.startswith("label")])
    datafiles = sorted([x for x in os.listdir(input_path) if x.startswith("data")])
    
    #only use the data where we have labels for
    datafiles = [x for x in datafiles if x.replace("data","labels") in labelfiles]
    
    #convert to numpy
    datafiles = np.asarray(datafiles)
    labelfiles = np.asarray(labelfiles)

    #PERMUTATION OF DATA
    np.random.seed(12345)
    shuffle_indices = np.random.permutation(len(datafiles))
    np.save("./shuffle_indices.npy", shuffle_indices)
    datafiles = datafiles[shuffle_indices]
    labelfiles = labelfiles[shuffle_indices]
    
    #Create train/validation/test split
    size = len(datafiles)
    trn_data = datafiles[:int(0.8 * size)]
    trn_labels = labelfiles[:int(0.8 * size)]
    tst_data = datafiles[int(0.8 * size):int(0.9 * size)]
    tst_labels = labelfiles[int(0.8 * size):int(0.9 * size)]
    val_data = datafiles[int(0.9 * size):]
    val_labels = labelfiles[int(0.9 * size):]
        
    return input_path, trn_data, trn_labels, val_data, val_labels, tst_data, tst_labels


class h5_input_reader(object):
    
    def __init__(self, path, channels):
        self.path = path
        self.channels = channels
    
    
    def read(self, datafile, labelfile):
        
        print(datafile)
        
        #set shape to none in the beginning
        shape = None
        
        #data
        with h5.File(self.path+'/'+datafile, "r") as f:
            shape = f['climate']['data'].shape
            data = np.expand_dims(f['climate']['data'][:,:,self.channels].astype(np.float32), axis=0)
        
        #label
        with h5.File(self.path+'/'+labelfile, "r") as f:
            label = np.expand_dims(f['climate']['labels'][...].astype(np.int32), axis=0)
        
        return data, label


def create_dataset(basepath, datafilelist, labelfilelist, batchsize, num_epochs, comm_size, comm_rank, channels, shuffle=False):
    #instantiate input reader
    h5r = h5_input_reader(basepath, channels)
    
    dataset = tf.data.Dataset.from_tensor_slices((datafilelist, labelfilelist))
    if comm_size>1:
        dataset = dataset.shard(comm_size, comm_rank)
    dataset = dataset.map(lambda dataname, labelname: tuple(tf.py_func(h5r.read, [dataname, labelname], [tf.float32, tf.int32])))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batchsize)
    dataset = dataset.repeat(num_epochs)
    
    return dataset


#main function
def main():
    #init horovod
    comm_rank = 0 
    comm_local_rank = 0
    comm_size = 1
    if horovod:
        hvd.init()
        comm_rank = hvd.rank() 
        comm_local_rank = hvd.local_rank()
        comm_size = hvd.size()
        print("Using distributed computation with Horovod: {} total ranks, I am rank {}".format(comm_size,comm_rank))
        
    #parameters
    batch = 4
    channels = [0,1,2,10]
    blocks = [3,3,4,4,7,7,10,10]
    num_epochs = 150
    
    #session config
    sess_config=tf.ConfigProto(inter_op_parallelism_threads=2,
                               intra_op_parallelism_threads=33,
                               log_device_placement=False,
                               allow_soft_placement=True)
    sess_config.gpu_options.visible_device_list = str(comm_local_rank)
    
    #get data
    training_graph = tf.Graph()
    print("Loading data...")
    path, trn_data, trn_labels, val_data, val_labels, tst_data, tst_labels = load_data()
    print("done.")
    
    with training_graph.as_default():
        #create datasets
        datafiles = tf.placeholder(tf.string, shape=[None])
        labelfiles = tf.placeholder(tf.string, shape=[None])
        trn_dataset = create_dataset(path, datafiles, labelfiles, batch, num_epochs, comm_size, comm_rank, channels, True)
        val_dataset = create_dataset(path, datafiles, labelfiles, batch, 1, comm_size, comm_rank, channels)
        
        #create iterators
        handle = tf.placeholder(tf.string, shape=[], name="iterator-placeholder")
        iterator = tf.data.Iterator.from_string_handle(handle, (tf.float32, tf.int32), 
                                                       ((batch, image_height, image_width, len(channels)), 
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
        logit, prediction = create_tiramisu(3, next_elem[0], image_height, image_width, len(channels), nb_layers_per_block=blocks, p=0.2, wd=1e-4)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=next_elem[1],logits=logit)
        global_step = tf.train.get_or_create_global_step()
        #set up optimizer
        opt = tf.train.RMSPropOptimizer(learning_rate=1e-3)
        if horovod:
            opt = hvd.DistributedOptimizer(opt)
        train_op = opt.minimize(loss, global_step=global_step)
        #set up streaming metrics
        labels_one_hot = tf.contrib.layers.one_hot_encoding(next_elem[1], 3)
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
        image_dir = './images'
        if comm_rank == 0:
            checkpoint_dir = './checkpoints'
            checkpoint_save_freq = num_steps_per_epoch
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
        #        sess.run(trn_init_op, feed_dict={handle: trn_handle, trn_feat_placeholder: trn, trn_lab_placeholder: trn_labels})
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
            sess.run(trn_init_op, feed_dict={handle: trn_handle, datafiles: trn_data, labelfiles: trn_labels})
            sess.run(val_init_op, feed_dict={handle: val_handle, datafiles: val_data, labelfiles: val_labels})

            #do the training
            epoch = 1
            train_loss = 0.
            while not sess.should_stop():
                
                #training loop
                try:
                    #construct feed dict
                    _, _, train_steps, tmp_loss = sess.run([train_op, iou_update_op, global_step, loss], feed_dict={handle: trn_handle})
                    train_steps_in_epoch = train_steps%num_steps_per_epoch
                    train_loss += tmp_loss
                    
                    if train_steps_in_epoch > 0:
                        #print step report
                        print("REPORT: rank {}, training loss for step {} (of {}) is {}".format(comm_rank, train_steps, num_steps, train_loss/train_steps_in_epoch))
                    else:
                        #print epoch report
                        train_loss /= num_steps_per_epoch
                        print("COMPLETED: rank {}, training loss for epoch {} (of {}) is {}".format(comm_rank, epoch, num_epochs, train_loss))
                        iou_score = sess.run(iou_op)
                        print("COMPLETED: rank {}, training IoU for epoch {} (of {}) is {}".format(comm_rank, epoch, num_epochs, iou_score))
                        
                        #evaluation loop
                        eval_loss = 0.
                        eval_steps = 0
                        while True:
                            try:
                                #construct feed dict
                                _, tmp_loss, val_model_predictions, val_model_labels = sess.run([iou_update_op, loss, prediction, next_elem[1]], feed_dict={handle: val_handle})
                                imsave(image_dir+'/test_pred_epoch'+str(epoch)+'_estep'
                                        +str(eval_steps)+'_rank'+str(comm_rank)+'.png',np.argmax(val_model_predictions[0,...],axis=2)*100)
                                imsave(image_dir+'/test_label_epoch'+str(epoch)+'_estep'
                                        +str(eval_steps)+'_rank'+str(comm_rank)+'.png',val_model_labels[0,...]*100)
                                eval_loss += tmp_loss
                                eval_steps += 1
                            except tf.errors.OutOfRangeError:
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
    main()
