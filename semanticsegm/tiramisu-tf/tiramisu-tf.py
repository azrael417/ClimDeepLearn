import tensorflow as tf
import tensorflow.contrib.keras as tfk
import numpy as np
import os
from scipy.misc import imsave

#horovod, yes or no?
horovod=True
try:
    import horovod.tensorflow as hvd
    print("Enabling Horovod Support")
except:
    horovod = False
    print("Disabling Horovod Support")

#GLOBAL CONSTANTS
image_height = 96
image_width = 144
comm_rank = 0
comm_local_rank = 0
comm_size = 1

def conv(x, nf, sz, wd, stride=1): 
    return tf.layers.conv2d(x, nf, sz, strides=(stride,stride), padding='same',
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
    #x = tf.layers.conv2d_transpose(inputs=x,strides=(2,2),kernel_size=(3,3),padding='same',filters=ch,kernel_initializer='he_uniform')
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



def create_tiramisu(nb_classes, img_input, nb_dense_block=6, 
         growth_rate=16, nb_filter=48, nb_layers_per_block=5, p=None, wd=0., training=True):
#def create_tiramisu(nb_classes, img_input, nb_dense_block=3, 
        #growth_rate=2, nb_filter=4, nb_layers_per_block=5, p=None, wd=0., training=True):
    
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)
    else: nb_layers = [nb_layers_per_block] * nb_dense_block

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
        _,r,c,f = x.get_shape().as_list()
    x = tf.reshape(x,[-1,image_height,image_width,nb_classes])
    return x, tf.nn.softmax(x)


def create_optimizer(loss, global_step, learning_rate=1e-3, use_horovod=True, LARS_nu=None, LARS_epsilon=1.0/16384.0):
    #set up optimizer
    opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    
    #allreduce the gradients if required:
    if use_horovod:
        opt = hvd.DistributedOptimizer(opt)
        
    #compute gradients
    grads_and_vars = opt.compute_gradients(loss)
    
    #do LARS hacking:
    if LARS_nu is not None and isinstance(LARS_nu, float):
        for idx, (g, v) in enumerate(grads_and_vars):
            if g is not None:
                v_norm = tf.norm(tensor=v, ord=2)
                g_norm = tf.norm(tensor=g, ord=2)
                lars_local_lr = tf.cond(
                                      pred = tf.logical_and( tf.not_equal(v_norm, tf.constant(0.0)), tf.not_equal(g_norm, tf.constant(0.0)) ),
                                      true_fn = lambda: LARS_nu * v_norm / g_norm,
                                      false_fn = lambda: LARS_epsilon)
                grads_and_vars[idx] = (tf.scalar_mul(lars_local_lr, g), v)
    
    #apply the gradients
    train_op = opt.apply_gradients(grads_and_vars, global_step=global_step)
    
    return train_op

#pass dictionary of steps and LR
def create_lr_schedule(rate_dictionary, global_step):
    schedule = sorted(rate_dictionary.items(), key=lambda x: x[0])
    boundaries = [int(x[0]) for x in schedule]
    rates = [x[1] for x in schedule]
    rates = rates[:1] + rates  # 
    assert len(boundaries) + 1 == len(rates)

    return tf.train.piecewise_constant(tf.cast(global_step, tf.int32), boundaries, rates)


#Load Data
def load_data():
    #Load the images and the labels
    imgs = np.load("/global/cscratch1/sd/tkurth/gb2018/tiramisu/tiny_set/images.npy").astype(np.float32)
    #imgs = np.load("/home/mudigonda/Data/tiramisu_clipped_combined_v1/images.npy").astype(np.float32)
    imgs = imgs.reshape([imgs.shape[0],imgs.shape[1],imgs.shape[2],1])
    labels = np.load("/global/cscratch1/sd/tkurth/gb2018/tiramisu/tiny_set/masks.npy").astype(np.int32)
    #labels = np.load("/home/mudigonda/Data/tiramisu_clipped_combined_v1/masks.npy").astype(np.int32)
    
    #Image metadata contains year, month, day, time_step, and lat/ lon data for each crop.  
    #See README in $SCRATCH/segmentation_labels/dump_v4 on CORI
    #image_metadata = np.load("/global/cscratch1/sd/tkurth/gb2018/tiramisu/small_set/image_metadata.npy")
    #imgs = np.load("/global/cscratch1/sd/tkurth/gb2018/tiramisu/tiny_set/images.npy").astype(np.float32)
    #imgs = imgs.reshape([imgs.shape[0],imgs.shape[1],imgs.shape[2],1])
    #labels = np.load("/global/cscratch1/sd/tkurth/gb2018/tiramisu/tiny_set/masks.npy").astype(np.int32)
    
    ##Image metadata contains year, month, day, time_step, and lat/ lon data for each crop.  
    ##See README in $SCRATCH/segmentation_labels/dump_v4 on CORI
    #image_metadata = np.load("/global/cscratch1/sd/tkurth/gb2018/tiramisu/tiny_set/image_metadata.npy")
    
    #do some slicing
    imgs = imgs[:,3:-3,...]
    labels = labels[:,3:-3,:]
    
    #DEBUG
    #only take slice:
    #imgs = imgs[:300,:]
    #labels = labels[:300,:]
    #image_metadata = image_metadata[:300,:]
    #DEBUG

    #PERMUTATION OF DATA
    np.random.seed(12345)
    shuffle_indices = np.random.permutation(len(imgs))
    np.save("./shuffle_indices.npy", shuffle_indices)
    imgs = imgs[shuffle_indices]
    labels = labels[shuffle_indices]
    #image_metadata = image_metadata[shuffle_indices]
    
    #Create train/validation/test split
    trn = imgs[:int(0.8*len(imgs))]
    trn_labels = labels[:int(0.8 * len(labels))]
    test = imgs[int(0.8*len(imgs)):int(0.9*len(imgs))]
    test_labels = labels[int(0.8*len(imgs)):int(0.9*len(imgs))]
    valid = imgs[int(0.9*len(imgs)):]
    valid_labels = labels[int(0.9*len(imgs)):]
    
    #length of data
    rnd_trn = len(trn_labels)
    rnd_test = len(test_labels)	
    
    #Normalize
    trn_mean = trn.mean(axis=0)
    trn_std = trn.std(axis=0)
    #trn = (trn - trn_mean)/trn_std
    #valid = (valid - trn_mean)/trn_std
    #test = (test - trn_mean)/trn_std
    
    return trn, trn_labels, valid, valid_labels, test, test_labels

#main function
def main():
    #init horovod
    if horovod:
        hvd.init()
        comm_rank = hvd.rank() 
        comm_local_rank = hvd.local_rank()
        comm_size = hvd.size()
        print("Using distributed computation with Horovod: {} total ranks, I am rank {}".format(comm_size,comm_rank))
    
    #parameters
    batch = 32
    blocks = [3,3,4,7,10]
    num_epochs = 150
    num_warmup_epochs = 1
    lr_train = 1e-3
    lr_warmup = 1e-5
    lars_nu = 0.001
    
    #get data
    training_graph = tf.Graph()
    print("Loading data...")
    trn, trn_labels, val, val_labels, tst, tst_labels = load_data()
    print("done.")
    
    with training_graph.as_default():
        #create datasets
        #placeholders
        trn_feat_placeholder = tf.placeholder(trn.dtype, trn.shape, name="train-feature-placeholder")
        trn_lab_placeholder = tf.placeholder(trn_labels.dtype, trn_labels.shape, name="train-labels-placeholder")
        val_feat_placeholder = tf.placeholder(val.dtype, val.shape, name="validation-feature-placeholder")
        val_lab_placeholder = tf.placeholder(val_labels.dtype, val_labels.shape, name="validation-labels-placeholder")
        tst_feat_placeholder = tf.placeholder(tst.dtype, tst.shape, name="test-feature-placeholder")
        tst_lab_placeholder = tf.placeholder(tst_labels.dtype, tst_labels.shape, name="test-labels-placeholder")
        #train dataset
        trn_dataset = tf.data.Dataset.from_tensor_slices((trn_feat_placeholder, trn_lab_placeholder))
        trn_dataset = trn_dataset.shard(comm_size, comm_rank)
        trn_dataset = trn_dataset.shuffle(buffer_size=100000)
        trn_dataset = trn_dataset.repeat(num_epochs)
        trn_dataset = trn_dataset.batch(batch)
        #validation dataset
        val_dataset = tf.data.Dataset.from_tensor_slices((val_feat_placeholder, val_lab_placeholder))
        val_dataset = val_dataset.shard(comm_size, comm_rank)
        val_dataset = val_dataset.repeat(1)
        val_dataset = val_dataset.batch(batch)
        #test dataset
        tst_dataset = tf.data.Dataset.from_tensor_slices((tst_feat_placeholder, tst_lab_placeholder))
        tst_dataset = tst_dataset.repeat(1)
        tst_dataset = tst_dataset.batch(batch)
        #create feedable iterator
        handle = tf.placeholder(tf.string, shape=[], name="iterator-placeholder")
        iterator = tf.data.Iterator.from_string_handle(handle, trn_dataset.output_types, trn_dataset.output_shapes)
        next_elem = iterator.get_next()
        #create init handles
        #tst
        trn_iterator = trn_dataset.make_initializable_iterator()
        trn_handle_string = trn_iterator.string_handle()
        trn_init_op = iterator.make_initializer(trn_dataset)
        #val
        val_iterator = val_dataset.make_initializable_iterator()
        val_handle_string = val_iterator.string_handle()
        val_init_op = iterator.make_initializer(val_dataset)
    
    
    #session config
    sess_config=tf.ConfigProto(inter_op_parallelism_threads=2,
                               intra_op_parallelism_threads=33,
                               log_device_placement=False,
                               allow_soft_placement=True)
    sess_config.gpu_options.visible_device_list = str(comm_local_rank)


    #create graph
    with training_graph.as_default():
        #compute epochs and stuff:
        num_samples = trn.shape[0] // comm_size
        num_steps_per_epoch = num_samples // batch
        num_steps = num_epochs*num_steps_per_epoch

        #set up model
        logit, prediction = create_tiramisu(3, next_elem[0], nb_layers_per_block=blocks, p=0.2, wd=1e-4)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=next_elem[1],logits=logit)
        global_step = tf.train.get_or_create_global_step()
        #set up 
        lr_schedule = create_lr_schedule({0: lr_warmup, (num_steps_per_epoch*num_warmup_epochs): lr_train}, global_step)
        #set up optimizer
        train_op = create_optimizer(loss, global_step, learning_rate=lr_schedule, use_horovod=horovod, LARS_nu=lars_nu)
        #encode labels
        labels_one_hot = tf.contrib.layers.one_hot_encoding(next_elem[1], 3)
        #iou metric
        iou_op, iou_update_op = tf.metrics.mean_iou(prediction,labels_one_hot,3,weights=None,metrics_collections=None,updates_collections=None,name="iou_score")

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
        trn_handle_string = trn_iterator.string_handle()
        val_handle_string = val_iterator.string_handle()
        with tf.train.MonitoredTrainingSession(config=sess_config, hooks=hooks) as sess:
            #initialize
            sess.run([init_op, init_local_op])
            #create iterator handles
            #trn_handle, val_handle = sess.run([trn_iterator.string_handle(), val_iterator.string_handle()])
            trn_handle, val_handle = sess.run([trn_handle_string, val_handle_string])
            #init iterators
            sess.run(trn_init_op, feed_dict={handle: trn_handle, trn_feat_placeholder: trn, trn_lab_placeholder: trn_labels})
            sess.run(val_init_op, feed_dict={handle: val_handle, val_feat_placeholder: val, val_lab_placeholder: val_labels})

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
                                imsave(image_dir+'/test_pred_epoch'+str(epoch)+'_estep'+str(eval_steps)+'_rank'+str(comm_rank)+'.png',np.argmax(val_model_predictions[0,...],axis=2)*100)
                                imsave(image_dir+'/test_label_epoch'+str(epoch)+'_estep'+str(eval_steps)+'_rank'+str(comm_rank)+'.png',val_model_labels[0,...]*100)
                                eval_loss += tmp_loss
                                eval_steps += 1
                            except tf.errors.OutOfRangeError:
                                eval_loss /= eval_steps
                                print("COMPLETED: rank {}, evaluation loss for epoch {} (of {}) is {}".format(comm_rank, epoch-1, num_epochs, eval_loss))
                                iou_score = sess.run(iou_op)
                                print("COMPLETED: rank {}, evaluation IoU for epoch {} (of {}) is {}".format(comm_rank, epoch-1, num_epochs, iou_score))
                                sess.run(val_init_op, feed_dict={handle: val_handle, val_feat_placeholder: val, val_lab_placeholder: val_labels})
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
