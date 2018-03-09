import tensorflow as tf
import numpy as np
import h5py as h5
import os
import time

#horovod, yes or no?
horovod=True
try:
    import horovod.tensorflow as hvd
except:
    horovod = False


#focal loss routine
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


#optimizer
def get_optimizer(opt_type, loss, global_step, learning_rate, momentum=0.):
    #set up optimizers
    if opt_type == "Adam":
        optim = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    elif opt_type == "RMSProp":
        optim = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    elif opt_type == "SGD":
        optim = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
    else:
        raise ValueError("Error, optimizer {} unsupported.".format(opt_type))
    
    #horovod wrapper
    if horovod:
        optim = hvd.DistributedOptimizer(optim)

    #return minimizer
    return optim.minimize(loss, global_step=global_step)
    

#larc optimizer:
def get_larc_optimizer(opt_type, loss, global_step, learning_rate, momentum=0., LARC_mode="clip", LARC_eta=0.002, LARC_epsilon=1.):
    #set up optimizers
    if opt_type == "Adam":
        optim = tf.train.RMSPropOptimizer(learning_rate=1.)
    elif opt_type == "RMSProp":
        optim = tf.train.RMSPropOptimizer(learning_rate=1.)
    elif opt_type == "SGD":
        optim = tf.train.MomentumOptimizer(learning_rate=1., momentum=momentum)
    else:
        raise ValueError("Error, optimizer {} unsupported.".format(opt_type))
        
    #horovod wrapper
    if horovod:
        optim = hvd.DistributedOptimizer(optim)
        
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
    return optim.apply_gradients(grads_and_vars, global_step=global_step)


#input reader class
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


#load data routine
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
