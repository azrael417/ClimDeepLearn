import tensorflow as tf
import numpy as np
import h5py as h5
import os
import time
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import linalg_ops
import multiprocessing
import signal

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


#neighborhood loss
def cluster_loss(predictions, ksize, padding="SAME", data_format="NHWC", name=None):
    r"""Computes average loss direction vectors around center points and then compute cosine similarity between center prediction and neighbors.
    That term can be added to a classification loss, for example a pointwise x-entropy to penalize salt-and-pepper noise or clusters smaller than kernel size.
    """
    if data_format=="NHWC":
        axis=-1
    elif data_format=="NCHW":
        axis=1
    else:
        raise ValueError("Error, format {} not supported for cluster_loss.".format(data_format))

    #compute average over neighborhood and normalize
    average_predictions = tf.nn.avg_pool(predictions, 
                                         ksize=[1,ksize,ksize,1], 
                                         strides=[1,1,1,1], 
                                         padding=padding,
                                         data_format=data_format,
                                         name=name)
    norm_average_predictions = tf.divide(average_predictions, tf.norm(average_predictions, ord=2, axis=axis, keepdims=True))
    norm_predictions = tf.divide(predictions, tf.norm(predictions, ord=2, axis=axis, keepdims=True))

    #compute scalar product across dim and reduce
    loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(norm_average_predictions,norm_predictions), axis=axis))
    
    return loss


#optimizer
def get_optimizer(opt_type, loss, global_step, learning_rate, momentum=0.):
    #set up optimizers
    if opt_type == "Adam":
        optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
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
def get_larc_optimizer(opt_type, loss, global_step, learning_rate, momentum=0., LARC_mode="clip", LARC_eta=0.002, LARC_epsilon=1./16000., gradient_lag=0):
    #set up optimizers
    if opt_type == "Adam":
        optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif opt_type == "RMSProp":
        optim = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    elif opt_type == "SGD":
        optim = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
    else:
        raise ValueError("Error, optimizer {} unsupported.".format(opt_type))

    # instead of using the horovod wrapper, we do the allreduce ourselves below
        
    #compute gradients
    grads_and_vars = optim.compute_gradients(loss)
    lag_ops = []
    for idx, (g, v) in enumerate(grads_and_vars):
        if g is not None:
            if gradient_lag > 0:
                g_lag = tf.Variable(initial_value=tf.zeros(g.shape, g.dtype),
                                    trainable=False,
                                    name=v.name.replace(":","_") + '_lag')
                g_next = g
                g = g_lag
                                    
            if horovod and (hvd.size() > 1):
                # if we ask for an average, it does a scalar divide, but
                #  we can bake that into the scaling below
                g = hvd.allreduce(g, average=False)
                g_scale = 1. / hvd.size()
            else:
                g_scale = 1

            v_norm = linalg_ops.norm(tensor=v, ord=2)
            g_norm = linalg_ops.norm(tensor=g, ord=2)

            larc_local_lr = control_flow_ops.cond(
                pred = math_ops.logical_and( math_ops.not_equal(v_norm, tf.constant(0.0)),
                                            math_ops.not_equal(g_norm, tf.constant(0.0)) ),
                                            true_fn = lambda: (LARC_eta * g_scale) * v_norm / g_norm,
                                            false_fn = lambda: LARC_epsilon)

            if LARC_mode=="scale":
                effective_lr = larc_local_lr
            else:
                effective_lr = math_ops.minimum(larc_local_lr, 1.0)

            effective_lr *= g_scale

            #multiply gradients
            g_scaled = math_ops.scalar_mul(effective_lr, g)
            grads_and_vars[idx] = (g_scaled, v)

            if gradient_lag > 0:
                # once we've computed g_scaled, it's safe to overwrite g_lag
                with tf.control_dependencies([g_scaled]):
                    lag_ops.append(g_lag.assign(g_next))

    #apply gradients, making sure to complete the forward pass first
    with tf.control_dependencies([loss]):
        grad_updates = optim.apply_gradients(grads_and_vars, global_step=global_step)
    if gradient_lag > 0:
        grad_updates = tf.group([ grad_updates ] + lag_ops)

    return grad_updates

class SharedExchangeBuffer(object):
    def __init__(self, count, size):
        self.count = count
        self.size = size
        self.arrays = [ multiprocessing.RawArray('B', size) for x in xrange(count) ]
        self.avail = set( xrange(count) )

    def get_free_slot(self):
        return self.avail.pop()

    def return_slot(self, slot):
        self.avail.add(slot)

    def pack_arrays(self, slot, *args):
        ofs = 0
        for a in args:
            #print 'packing', a.dtype, a.shape
            view = np.frombuffer(self.arrays[slot], dtype=a.dtype, count=a.size, offset=ofs).reshape(a.shape)
            ofs += a.nbytes
            view[...] = a
        return tuple((a.dtype, a.size, a.shape) for a in args)

    def unpack_arrays(self, slot, *args):
        ofs = 0
        results = []
        for a in args:
            #print 'unpacking', a
            view = np.frombuffer(self.arrays[slot], dtype=a[0], count=a[1], offset=ofs).reshape(a[2])
            ofs += view.nbytes
            results.append(view[...])
        return tuple(results)

smem = SharedExchangeBuffer(4, 128 << 20)

# defined outside of the h5_input_reader class due to weirdness with pickling
#  class methods
def _h5_input_subprocess_reader(path, channels, weights, minvals, maxvals, update_on_read, dtype, sample_target, shared_slot):
    #begin_time = time.time()
    with h5.File(path, "r", driver="core", backing_store=False, libver="latest") as f:
        #get min and max values and update stored values
        if update_on_read:
            # stats order is mean, max, min, stddev
            minvals = np.minimum(minvals, f['climate']['stats'][channels,2])
            maxvals = np.maximum(maxvals, f['climate']['stats'][channels,1])

        #get data
        if 'channels' in f['climate']:
            # some channels have been dropped from the file, so map to the
            #  actual locations in the file array
            channel_list = list(f['climate']['channels'])
            channels = [ channel_list.index(c) for c in channels ]

        data = f['climate']['data'][channels,:,:].astype(np.float32)

        #do min/max normalization
        for c in range(len(channels)):
            data[c,:,:] = (data[c,:,:]-minvals[c])/(maxvals[c]-minvals[c])

        #get label
        label = f['climate']['labels'][...]

    # cast data and labels if needed
    if data.dtype != dtype:
        data = data.astype(dtype)

    if label.dtype != np.int32:
        label = label.astype(np.int32)
        
    if sample_target is not None:
        # determine the number of pixels in each of the three classes
        counts = np.histogram(label, bins=[0,1,2,3])[0]
        # assign a per-class probability that delivers the target number of
        #  pixels of each class (in expectation) in the mask
        prob = float(sample_target) / counts
        # randomly select actual pixels to include in the loss function on
        #  this step
        r = np.random.uniform(size=label.shape)
        weights = (r < prob[label]).astype(dtype)
    else:
        #get weights - choose per-channel based on the labels
        weights = weights[label]

    #time
    #end_time = time.time()
    #print "%d: Time to read image %.3f s" % (os.getpid(), end_time-begin_time)
    data, label, weights = smem.pack_arrays(shared_slot, data, label, weights)
    return data, label, weights, minvals, maxvals

#input reader class
class h5_input_reader(object):
    
    def __init__(self, path, channels, weights, dtype, normalization_file=None, update_on_read=False, sample_target=None):
        self.path = path
        self.channels = channels
        self.update_on_read = update_on_read
        self.dtype = dtype.as_numpy_dtype()
        self.weights = np.asarray(weights, dtype=self.dtype)
        self.sample_target = sample_target
        if normalization_file:
             with h5.File(self.path+'/'+normalization_file, "r", libver="latest") as f:
                 # stats order is mean, max, min, stddev
                 self.minvals = f['climate']['stats'][self.channels,2].astype(np.float32)
                 self.maxvals = f['climate']['stats'][self.channels,1].astype(np.float32)
        else:
            self.minvals = np.asarray([np.inf]*len(channels), dtype=np.float32)
            self.maxvals = np.asarray([-np.inf]*len(channels), dtype=np.float32)

    # suppress SIGINT when we launch pool so ^C's go to main process
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = multiprocessing.Pool(processes=4)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    def read(self, datafile):
        path = self.path+'/'+datafile
        #begin_time = time.time()
        #nvtx.RangePush('h5_input', 8)
        shared_slot = smem.get_free_slot()
        data, label, weights, new_minvals, new_maxvals = self.pool.apply(_h5_input_subprocess_reader, (path, self.channels, self.weights, self.minvals, self.maxvals, self.update_on_read, self.dtype, self.sample_target, shared_slot))
        if self.update_on_read:
            self.minvals = np.minimum(self.minvals, new_minvals)
            self.maxvals = np.maximum(self.maxvals, new_maxvals)
        data, label, weights = smem.unpack_arrays(shared_slot, data, label, weights)
        smem.return_slot(shared_slot)
        #nvtx.RangePop()
        #end_time = time.time()
        #print "Time to read %s = %.3f s" % (path, end_time-begin_time)
        return data, label, weights

    def sequential_read(self, datafile):
        
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
def load_data(input_path, max_files):
    #look for labels and data files
    files = sorted([x for x in os.listdir(input_path) if x.startswith("data")])

    #we will choose to load only the first p files
    if max_files > 0:
        files = files[:max_files]

    #convert to numpy
    files = np.asarray(files)

    #PERMUTATION OF DATA
    np.random.seed(12345)
    shufflefile = "./shuffle_indices.npy"
    if not os.path.isfile(shufflefile):
        shuffle_indices = np.random.permutation(len(files))
        if hvd.rank() == 0:
            np.save(shufflefile,shuffle_indices)
    else:
        try:
            shuffle_indices = np.load(shufflefile)
        except:
            shuffle_indices = np.random.permutation(len(files))
    files = files[shuffle_indices]

    #Create train/validation/test split
    size = len(files)
    trn_data = files[:int(0.8 * size)]
    tst_data = files[int(0.8 * size):int(0.9 * size)]
    val_data = files[int(0.9 * size):]

    return trn_data, val_data, tst_data


#load model wrapper
def load_model(sess, saver, checkpoint_dir):
    print("Looking for model in {}".format(checkpoint_dir))
    #get list of checkpoints
    checkpoints = [x.replace(".index","") for x in os.listdir(checkpoint_dir) if x.startswith("model.ckpt") and x.endswith(".index")]
    checkpoints = sorted([(int(x.split("-")[1]),x) for x in checkpoints], key=lambda tup: tup[0])
    latest_ckpt = os.path.join(checkpoint_dir,checkpoints[-1][1])
    print("Restoring model {}".format(latest_ckpt))
    try:
        saver.restore(sess, latest_ckpt)
        print("Model restoration successful.")
    except:
        print("Loading model failed, starting fresh.")
