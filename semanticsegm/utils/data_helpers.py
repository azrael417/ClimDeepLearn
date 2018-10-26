import tensorflow as tf
import numpy as np

#colormap for plotting                                       label predict color
plot_colormap = np.array([[[  0,  0,  0],  #   0      0     black
                      [255,  0,255],       #   0      1     purple
                      [  0,255,255]],      #   0      2     cyan
                     [[  0,255,  0],       #   1      0     green
                      [128,128,128],       #   1      1     grey
                      [255,255,  0]],      #   1      2     yellow
                     [[255,  0,  0],       #   2      0     red
                      [  0,  0,255],       #   2      1     blue
                      [255,255,255]],      #   2      2     white
                     ])

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


def create_dataset(h5ir, datafilelist, batchsize, num_epochs, comm_size, comm_rank, dtype, shuffle=False):
    if comm_size > 1:
        # use an equal number of files per shard, leaving out any leftovers
        per_shard = len(datafilelist) // comm_size
        sublist = datafilelist[0:per_shard * comm_size]
        dataset = tf.data.Dataset.from_tensor_slices(sublist)
        dataset = dataset.shard(comm_size, comm_rank)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(datafilelist)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.map(map_func=lambda dataname: tuple(tf.py_func(h5ir.read, [dataname], [dtype, tf.int32, dtype, tf.string])),
                          num_parallel_calls = 4)
    dataset = dataset.prefetch(16)
    # make sure all batches are equal in size
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batchsize))
    dataset = dataset.repeat(num_epochs)

    return dataset
