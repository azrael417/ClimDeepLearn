import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops

np.random.seed(12345)

frequencies = [0.982,0.00071,0.017]
weights = np.sqrt([1./x for x in frequencies])
weights /= np.sum(weights)

ww = np.zeros((1,1,1,len(weights)))
ww[0,0,0,:] = weights[:]
weights_ph = tf.constant(ww, dtype=tf.float32)

label = np.random.randint(0,3,size=(1,3,3))
logit = np.random.uniform(size=(1,3,3,3))

label_ph = tf.placeholder(tf.int32,shape=(1,3,3),name='label')
logit_ph = tf.placeholder(tf.float32,shape=(1,3,3,3),name = 'logit')
pred_ph = tf.nn.softmax(logit_ph)
labels_one_hot = tf.contrib.layers.one_hot_encoding(label_ph, 3)
#weighted_labels_one_hot = tf.multiply(labels_one_hot, weights_ph)
gamma = 2.
weighted_labels_one_hot = tf.multiply(tf.multiply(labels_one_hot, (1-pred_ph)**gamma),weights_ph)
loss = tf.losses.softmax_cross_entropy(onehot_labels=weighted_labels_one_hot,logits=logit_ph)


def focal_loss(prediction_tensor, target_tensor, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = softmax(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    #subtract the mean before softmaxing
    #softmax_p = tf.nn.softmax( prediction_tensor - tf.reduce_mean( prediction_tensor, axis=3 ), axis=3 )
    softmax_p = tf.nn.softmax( prediction_tensor,axis=3) 
    zeros = array_ops.zeros_like(softmax_p, dtype=softmax_p.dtype)
    pos_p_sub = array_ops.where(target_tensor >= softmax_p, target_tensor - softmax_p, zeros)
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, softmax_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(softmax_p, 1e-8, 1.)) \
                          - (1. - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1. - softmax_p, 1e-8, 1.))
    return tf.reduce_mean(tf.reduce_sum(per_entry_cross_ent,axis=3))

focal_loss_l = focal_loss(labels_one_hot, logit_ph,gamma=0.,alpha=.5)

with tf.Session() as sess:
    l, loh, wloh,pred,focal_loss_ll = sess.run([loss,labels_one_hot,weighted_labels_one_hot,pred_ph,focal_loss_l],feed_dict={logit_ph:logit,label_ph:label})
    print("loss is {}, labels one hot is {}, weighted labels one hot is {}".format(l,loh,wloh))
    print("labels are {}".format(label))
    print("weights are {}".format(weights))
    print("predictions are {}".format(pred))
    print("Numpy loss is {}".format(np.mean(np.sum(np.log(pred)*(-wloh),axis=3))))
    #print("Numpy loss unweighted is {}".format(np.mean(np.sum(np.log(pred)*(-loh),axis=3))))
    #print("Numpy loss unweighted w clipping is {}".format(np.mean(np.sum(np.log(np.clip(pred,1e-8,1.))*(-loh),axis=3))))
    print("Focal loss is {}".format(focal_loss_ll))
