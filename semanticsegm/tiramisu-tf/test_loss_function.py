import tensorflow as tf
import numpy as np

np.random.seed(12345)

frequencies = [0.982,0.00071,0.017]
weights = np.sqrt([1./x for x in frequencies])
weights /= np.sum(weights)

ww = np.zeros((1,1,1,len(weights)))
ww[0,0,0,:] = weights[:]
weights_ph = tf.constant(ww, dtype=tf.float32)

label = np.random.randint(0,3,size=(1,3,3))
logit = np.random.uniform(size=(1,3,3,3))
print(label.shape)
print(logit.shape)

label_ph = tf.placeholder(tf.int32,shape=(1,3,3),name='label')
logit_ph = tf.placeholder(tf.float32,shape=(1,3,3,3),name = 'logit')
pred_ph = tf.nn.softmax(logit_ph)
labels_one_hot = tf.contrib.layers.one_hot_encoding(label_ph, 3)
weighted_labels_one_hot = tf.multiply(labels_one_hot, weights_ph)
loss = tf.losses.softmax_cross_entropy(onehot_labels=weighted_labels_one_hot,logits=logit_ph)

with tf.Session() as sess:
    l, loh, wloh,pred = sess.run([loss,labels_one_hot,weighted_labels_one_hot,pred_ph],feed_dict={logit_ph:logit,label_ph:label})
    print("loss is {}, labels one hot is {}, weighted labels one hot is {}".format(l,loh,wloh))
    print("labels are {}".format(label))
    print("weights are {}".format(weights))
    print("predictions are {}".format(pred))
    print("Numpy loss is {}".format(np.mean(np.sum(np.log(pred)*(-wloh),axis=3))))
