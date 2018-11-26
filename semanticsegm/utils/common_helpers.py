

import sys
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import linalg_ops

#horovod, yes or no?
try:
    import horovod.tensorflow as hvd
except:
    print("Warning, horovod not installed")


#region profiling hook for cuda
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


#typecast wrapper
def ensure_type(input, dtype):
    if input.dtype != dtype:
        return tf.cast(input, dtype)
    else:
        return input

#return parameters in model
def get_number_of_trainable_parameters():
    result = np.sum([int(np.prod(v.shape)) for v in tf.trainable_variables()])
    return result


#helper routine for FP16<->FP32 conversions
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
        variable = ensure_type(variable, dtype)
    return variable


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


#helper func
def get_dict_default(dictionary,varname,default):
    var=default
    if varname in dictionary:
        var=dictionary[varname]
    deftype=type(default)
    return deftype(var)


#learning rate
def get_learning_rate(optimizer, global_step, steps_per_epoch):
    with tf.device("/device:CPU:0"):
        learning_rate=tf.constant(get_dict_default(optimizer,"learning_rate",1.e-4),
                                  dtype=tf.float32)

        lr_decay_mode = get_dict_default(optimizer, "lr_decay", "none")
        if lr_decay_mode == "none":
            pass
        elif lr_decay_mode.startswith("poly:"):
            _, power, max_epoch = lr_decay_mode.split(":")
            # tf.train.polynomial_decay doesn't have a staircase mode, so
            #  implement it ourselves
            global_epoch = tf.floordiv(global_step, steps_per_epoch)
            learning_rate = tf.train.polynomial_decay(learning_rate=learning_rate,
                                                      global_step=global_epoch,
                                                      decay_steps=int(max_epoch),
                                                      end_learning_rate=0,
                                                      power=float(power))
        elif lr_decay_mode.startswith("exp:"):
            args = lr_decay_mode.split(":")
            rate = float(args[1])
            epochs = int(args[2]) if (len(args) > 2) else 1
            learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                                       global_step=global_step,
                                                       decay_steps=(steps_per_epoch * epochs),
                                                       decay_rate=rate,
                                                       staircase=True)
        elif lr_decay_mode.startswith("piece:"):
            args = lr_decay_mode.split(":")
            assert (len(args) % 2) == 1
            global_epoch = tf.floordiv(global_step, steps_per_epoch)
            prev_scale = lambda: 1.0
            cases = []
            for i in range(1,len(args),2):
                epoch_cutoff = int(args[i])
                cases.append((tf.less(global_epoch, epoch_cutoff),
                              prev_scale))
                lr_scale = float(args[i+1])
                # minor shenanigans to capture by value in lambda
                prev_scale = (lambda x: lambda: x)(lr_scale)
            learning_rate *= tf.case(pred_fn_pairs=cases,
                                     default=prev_scale,
                                     exclusive=False)
        else:
            print("ERROR: Unknown lr_decay mode:", lr_decay_mode)
            exit(1)

    return learning_rate

#optimizer
def get_optimizer(optimizer, loss, global_step, steps_per_epoch, use_horovod):
    #get learning rate
    learning_rate=get_learning_rate(optimizer, global_step, steps_per_epoch)

    #set up optimizers
    opt_type=get_dict_default(optimizer,"opt_type","Adam")

    #switch optimizer
    if opt_type == "Adam":
        beta1=get_dict_default(optimizer,"beta1",0.9)
        beta2=get_dict_default(optimizer,"beta2",0.999)
        optim = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)
    elif opt_type == "RMSProp":
        decay=get_dict_default(optimizer,"decay",0.9)
        momentum=get_dict_default(optimizer,"momentum",0.)
        centered=get_dict_default(optimizer,"centered",False)
        optim = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay, momentum=momentum, centered=centered)
    elif opt_type == "SGD":
        momentum=get_dict_default(optimizer,"momentum",0.)
        optim = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
    else:
        raise ValueError("Error, optimizer {} unsupported.".format(opt_type))

    #horovod wrapper
    if use_horovod:
        optim = hvd.DistributedOptimizer(optim)

    #return minimizer
    return optim.minimize(loss, global_step=global_step), learning_rate


#larc optimizer:
def get_larc_optimizer(optimizer, loss, global_step, steps_per_epoch, use_horovod):
    #get learning rate
    learning_rate = get_learning_rate(optimizer, global_step, steps_per_epoch)

    #get LARC stuff
    LARC_mode = get_dict_default(optimizer,"LARC_mode","clip")
    LARC_eta = get_dict_default(optimizer,"LARC_eta",0.002)
    LARC_epsilon = get_dict_default(optimizer,"LARC_epsilon",1./16000.)

    #lag
    gradient_lag = get_dict_default(optimizer,"gradient_lag",0)

    #set up optimizers
    opt_type = get_dict_default(optimizer,"opt_type","LARC-Adam")

    #set up optimizers
    if opt_type == "LARC-Adam":
        beta1 = get_dict_default(optimizer,"beta1",0.9)
        beta2 = get_dict_default(optimizer,"beta2",0.999)
        optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif opt_type == "LARC-RMSProp":
        optim = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    elif opt_type == "LARC-SGD":
        momentum = get_dict_default(optimizer,"momentum",0.)
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

            if use_horovod and (hvd.size() > 1):
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
                                            true_fn = lambda: (LARC_eta / g_scale) * v_norm / g_norm,
                                            false_fn = lambda: LARC_epsilon)

            if LARC_mode=="scale":
                effective_lr = larc_local_lr
            else:
                # DEBUG
                #effective_lr = math_ops.minimum(larc_local_lr, 1.0)
                #we need to see which LR to take and then divide out the LR because otherwise it will be multiplied in
                #again when we apply the gradients
                effective_lr = math_ops.minimum(larc_local_lr, learning_rate) / learning_rate
                # DEBUG

            # rescale gradients
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

    return grad_updates, learning_rate
