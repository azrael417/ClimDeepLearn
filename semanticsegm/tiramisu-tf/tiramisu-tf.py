import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np

#GLOBAL CONSTANTS
image_height = 96
image_width = 144


def conv(x, nf, sz, wd, stride=1): 
    return tf.layers.conv2d(x, nf, sz, strides=(stride,stride), padding='same',
                            #kernel_initializer='he_uniform',
                            kernel_initializer= tf.initializers.random_uniform(),
                            bias_initializer=tf.initializers.random_uniform(),
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
				   kernel_initializer=tf.initializers.random_uniform(),
				   bias_initializer=tf.initializers.random_uniform())
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
	x = tf.nn.softmax(x)
    return x


#Load Data
def load_data():
    #Load the images and the labels
    imgs = np.load("/global/cscratch1/sd/tkurth/gb2018/tiramisu/small_set/images.npy").astype(np.float32)
    imgs = imgs.reshape([imgs.shape[0],imgs.shape[1],imgs.shape[2],1])
    labels = np.load("/global/cscratch1/sd/tkurth/gb2018/tiramisu/small_set/masks.npy").astype(np.int32)
    
    #Image metadata contains year, month, day, time_step, and lat/ lon data for each crop.  
    #See README in $SCRATCH/segmentation_labels/dump_v4 on CORI
    image_metadata = np.load("/global/cscratch1/sd/tkurth/gb2018/tiramisu/small_set/image_metadata.npy")
    
    #do some slicing
    imgs = imgs[:,3:-3,...]
    labels = labels[:,3:-3,:]
    
    #split by rank:
    num_samples_per_rank = imgs.shape[0] // hvd.size()
    start = hvd.rank() * num_samples_per_rank
    end = np.min([(hvd.rank()+1) * num_samples_per_rank, imgs.shape[0]])
    imgs = imgs[start:end,:]
    labels = labels[start:end,:]

    #PERMUTATION OF DATA
    np.random.seed(12345)
    shuffle_indices = np.random.permutation(len(imgs))
    np.save("./shuffle_indices.npy", shuffle_indices)
    imgs = imgs[shuffle_indices]
    labels = labels[shuffle_indices]
    image_metadata = image_metadata[shuffle_indices]
    
    #Create train/validation/test split
    trn = imgs[:int(0.8*len(imgs))]
    trn_labels = labels[:int(0.8 * len(labels))]
    test = imgs[int(0.8*len(imgs)):int(0.9*len(imgs))]
    test_labels = labels[int(0.8*len(imgs)):int(0.9*len(imgs))]
    valid = imgs[int(0.9*len(imgs)):]
    valid_labels = labels[int(0.9*len(imgs)):]
    
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
    #parameters
    batch = 32
    blocks = [3,3,4,7,10]
    num_epochs = 10

    #init horovod
    hvd.init()
    
    #get data
    training_graph = tf.Graph()
    print("Loading data...")
    trn, trn_labels, valid, valid_labels, test, test_labels = load_data()
    print("done.")
    
    with training_graph.as_default():
        #create datasets
        #training
        feat_placeholder = tf.placeholder(trn.dtype, trn.shape, name="feature-placeholder")
        lab_placeholder = tf.placeholder(trn_labels.dtype, trn_labels.shape, name="labels-placeholder")
        trn_dataset = tf.data.Dataset.from_tensor_slices((feat_placeholder, lab_placeholder))
        trn_dataset = trn_dataset.shuffle(buffer_size=10000)
        trn_dataset = trn_dataset.repeat(num_epochs)
        trn_dataset = trn_dataset.batch(batch)
        #validation
        val_dataset = tf.data.Dataset.from_tensor_slices((feat_placeholder, lab_placeholder))
        val_dataset = val_dataset.repeat(1)
        val_dataset = val_dataset.batch(batch)
        #create feedable iterator
        handle = tf.placeholder(tf.string, shape=[], name="iterator-placeholder")
        iterator = tf.data.Iterator.from_string_handle(handle, trn_dataset.output_types, trn_dataset.output_shapes)
        next_elem = iterator.get_next()
        #create init handles
        trn_iterator = trn_dataset.make_initializable_iterator()
        val_iterator = val_dataset.make_initializable_iterator()
        trn_init_op = iterator.make_initializer(trn_dataset)
        val_init_op = iterator.make_initializer(val_dataset)
    
    #session config
    sess_config=tf.ConfigProto(inter_op_parallelism_threads=2,
                               intra_op_parallelism_threads=33,
                               log_device_placement=False,
                               allow_soft_placement=True)
    sess_config.gpu_options.visible_device_list = str(hvd.local_rank())


    #create graph
    with training_graph.as_default():
        #set up model
        #images = tf.placeholder(tf.float32, [None, trn.shape[1], trn.shape[2], 1])
        #labels = tf.placeholder(tf.int32, [None, trn.shape[1], trn.shape[2], 1])
        model = create_tiramisu(3, next_elem[0])
        loss = tf.losses.sparse_softmax_cross_entropy(labels=next_elem[1],logits=model)
        global_step = tf.train.get_or_create_global_step()
        #set up optimizer
        train_op = tf.train.RMSPropOptimizer(learning_rate=1e-3)
        train_op = hvd.DistributedOptimizer(train_op)
        train_op = train_op.minimize(loss)
        
        #compute epochs and stuff:
        num_samples = trn.shape[0]
        num_batches_per_epoch = num_samples // batch
        num_steps = num_epochs*num_batches_per_epoch
        
        #hooks
        #these hooks are essential
        hooks = [hvd.BroadcastGlobalVariablesHook(0), tf.train.StopAtStepHook(last_step=num_steps)]
        #initializers:
        init_op =  tf.global_variables_initializer()
        
        #checkpointing
        if hvd.rank() == 0:
            checkpoint_dir = './checkpoints' if hvd.rank() == 0 else None
            checkpoint_save_freq = num_batches_per_epoch
            checkpoint_saver = tf.train.Saver(max_to_keep = 1000)
            hooks.append(tf.train.CheckpointSaverHook(checkpoint_dir=checkpoint_dir, save_steps=checkpoint_save_freq, saver=checkpoint_saver))
        
        #start session
        with tf.train.MonitoredTrainingSession(config=sess_config, hooks=hooks) as sess:
            #initialize
            sess.run(init_op)
            #iterators
            #create handles
            trn_handle, val_handle = sess.run([trn_iterator.string_handle(), val_iterator.string_handle()])
            #init iterators
            sess.run(trn_init_op, feed_dict={handle: trn_handle, feat_placeholder: trn,lab_placeholder: trn_labels})

            #do the training
            epoch = 1
            mean_loss = 0.
            train_steps = 0
            while not sess.should_stop():
                try:
                    #construct feed dict
                    _, tmp_loss = sess.run([train_op,loss], feed_dict={handle: trn_handle})
                    train_steps += 1
                    mean_loss += tmp_loss
                    print("Loss for step {} (of {}) is {}".format(train_steps, num_batches_per_epoch, mean_loss/train_steps))
                except tf.errors.OutOfRangeError:
                    mean_loss /= train_steps
                    print("Loss for epoch {} (of {}) is {}".format(epoch, num_epochs, mean_loss))
                    #reinitialize dataset
                    sess.run(trn_init_op, feed_dict={handle: trn_handle, feat_placeholder: trn,lab_placeholder: trn_labels})
                    #reset counters
                    epoch += 1
                    train_steps = 0
                    mean_loss = 0.


if __name__ == '__main__':
    main()
