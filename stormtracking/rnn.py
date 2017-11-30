from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.contrib import rnn
from inference import *

w=129; h=86;
display_step=10;
testing_step=100;
training_steps = 200000
input_size =timesteps=24;
feature_size=w*h;
# Network Parameters
lstm_size=num_hidden = 170 # hidden layer num of features
num_classes = 1 # Length of output
number_of_layers=1; #Start from only one layer


def lstm_cell():
  return tf.contrib.rnn.BasicLSTMCell(lstm_size)
stacked_lstm=tf.contrib.rnn.MultiRNNCell(
    [lstm_cell() for _ in range(number_of_layers)])

initial_state = state = stacked_lstm.zero_state(FLAGS.batch_size, tf.float32)

# Define weights state output
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Define weights localization output
weights2 = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes*2]))
}
biases2 = {
    'out': tf.Variable(tf.random_normal([num_classes*2]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Get lstm cell output
    print(stacked_lstm)
    print(x)
    print(initial_state)
    outputs, states = tf.contrib.rnn.static_rnn(stacked_lstm, x, initial_state);

    output_logits=[]; output_lonlat=[];
    # Linear activation, using rnn inner loop last output
    for t in range(timesteps):
        output_logits.append(tf.matmul(outputs[t], weights['out']) + biases['out'])
        output_lonlat.append(tf.matmul(outputs[t],weights2['out']+biases2['out']));
    #output_logits=tf.tanh(output_logits)
    output_logits=tf.nn.softmax(output_logits)
    return output_logits,output_lonlat


