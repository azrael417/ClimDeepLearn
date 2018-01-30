import tensorflow as tf
import numpy as np

#GLOBAL CONSTANTS
image_height = 96
image_width = 144


def conv(x, nf, sz, wd, stride=1): 
    return tf.layers.conv2d(x, nf, sz, strides=(stride,stride), padding='same',
                            #kernel_initializer='he_uniform',
                            #bias_initializer=tf.zeros_zeros_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=wd)
                            )

def dense_block(n, x, growth_rate, p, wd, training):

    added = []
    for i in range(n):
        with tf.name_scope("bn_relu_conv%i"%i) as scope:
            b = tf.layers.batch_normalization(x, training=training)
            b = tf.nn.relu(b)
            b = conv(b, growth_rate, sz=3, wd=wd)
            if p: b = tf.layers.dropout(b, p=p, training=training)

            x = tf.concat([x, b], axis=-1)
            added.append(b)

    return x, added

def transition_dn(x, p, wd, training):
    with tf.name_scope("bn_relu_conv") as scope:
        b = tf.layers.batch_normalization(x, training=training)
        b = tf.nn.relu(b)
        b = conv(b, x.get_shape().as_list()[-1], sz=1, wd=wd, stride=2)
        if p: b = tf.layers.dropout(b, p=p, training=training)
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
    x = tf.layers.conv2d_transpose(inputs=x,strides=(2,2),kernel_size=(3,3),padding='same',filters=ch)
    return x 
    
	
def up_path(added,skips,nb_layers,growth_rate,p,wd,training):
    for i,n in enumerate(nb_layers):
	x = transition_up(added,wd,training)
	x = tf.concat([x,skips[i]],axis=-1)
	x, added = dense_block(n,x,growth_rate,p,wd,training=training)
    return x

# def create_tiramisu(nb_classes, img_input, nb_dense_block=6, 
        # growth_rate=16, nb_filter=48, nb_layers_per_block=5, p=None, wd=0., training=True):
def create_tiramisu(nb_classes, img_input, nb_dense_block=3, 
        growth_rate=2, nb_filter=4, nb_layers_per_block=5, p=None, wd=0., training=True):
    
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
	imgs = np.load("/home/mudigonda/Data/tiramisu_clipped_combined_v2/images.npy")
	imgs = imgs.reshape([imgs.shape[0],imgs.shape[1],imgs.shape[2],1])
	labels = np.load("/home/mudigonda/Data/tiramisu_clipped_combined_v2/masks.npy")

	#Image metadata contains year, month, day, time_step, and lat/ lon data for each crop.  
	#See README in $SCRATCH/segmentation_labels/dump_v4 on CORI
	image_metadata = np.load("/home/mudigonda/Data/tiramisu_clipped_combined_v2/image_metadata.npy")

	imgs = imgs[:,3:-3,...]
	labels = labels[:,3:-3,:]

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

def main():

    training_graph = tf.Graph()
    trn, trn_labels, valid, valid_labels, test, test_labels = load_data()
    batch = 32
    #trn = np.random.randint(255,size=(10,96,144,1))
    #trn_labels = np.random.randint(3,size=(10,96,144,1))
    #trn = (trn - trn.mean(axis=0))/trn.std(axis=0)
    

    with training_graph.as_default():

        with tf.Session() as sess:
            images = tf.placeholder(tf.float32, [None, trn.shape[1], trn.shape[2], 1])
	    labels = tf.placeholder(tf.int32, [None, trn.shape[1], trn.shape[2], 1])
            model = create_tiramisu(3, images)
	    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=model)
	    train_op = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(loss)
	    tf.global_variables_initializer().run()
	    indices = np.random.choice(trn.shape[0],batch)
	    #batch_trn = trn[indices, ...].reshape([batch,96,144,1])
	    #batch_trn_labels = trn_labels[:batch,...].reshape([batch,96,144,1])
	    #batch_trn_labels = trn_labels[indices, ...].reshape([batch,96,144,1])
	    for ii in range(100):
		l = []
		for jj in np.arange(0,trn.shape[0],batch):
		    try:
			batch_trn = trn[jj:jj+batch,...]
			batch_trn_labels = trn_labels[jj:jj+batch,...] 
		    	batch_trn = batch_trn.reshape([batch,96,144,1])
		    	batch_trn_labels = batch_trn_labels.reshape([batch,96,144,1])
		    except:
			batch_trn = trn[jj:,...]
			batch_trn_labels = trn_labels[jj:,...] 
		    	batch_trn = batch_trn.reshape([batch_trn.shape[0],96,144,1])
		    	batch_trn_labels = batch_trn_labels.reshape([batch_trn_labels.shape[0],96,144,1])
		    feed_dict = {images:batch_trn,labels:batch_trn_labels}
	            _,loss_itr = sess.run([train_op,loss],feed_dict=feed_dict)
		    l.append(loss_itr)
		    if np.mod(jj,640):
			    print("Loss for epoch {} and iteration {} is {}".format(ii,jj,loss_itr))
	        print("##########Loss for epoch {} is {}#########".format(ii,np.array(l).mean()))
	    import IPython; IPython.embed()
            writer = tf.summary.FileWriter('./logs/dev', sess.graph)

if __name__ == '__main__':
    main()
