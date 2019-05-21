import numpy as np
import tensorflow as tf
import h5py as h5
import os
import time
import multiprocessing
import signal

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
    dataset = dataset.map(map_func=lambda dataname: tuple(tf.py_func(h5ir.read, [dataname, False], [dtype, tf.int32, dtype, tf.string])),
                          num_parallel_calls = 4)
    dataset = dataset.prefetch(16)
    # make sure all batches are equal in size
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batchsize))
    dataset = dataset.repeat(num_epochs)

    return dataset


class SharedExchangeBuffer(object):
    def __init__(self, count, size):
        self.count = count
        self.size = size
        self.arrays = [ multiprocessing.RawArray('B', size) for x in range(count) ]
        self.avail = set( range(count) )

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
            #important to copy the data here
            view[...] = a[...]
        return tuple((a.dtype, a.size, a.shape) for a in args)

    def unpack_arrays(self, slot, *args):
        ofs = 0
        results = []
        for a in args:
            #print 'unpacking', a
            view = np.frombuffer(self.arrays[slot], dtype=a[0], count=a[1], offset=ofs).reshape(a[2])
            ofs += view.nbytes
            #important to copy the data here
            arr = view.copy()
            results.append(arr)
        return tuple(results)

#global shared memory buffer
smem = SharedExchangeBuffer(4, 128 << 20)

# defined outside of the h5_input_reader class due to weirdness with pickling
#  class methods
def _h5_input_subprocess_reader(path, channels, weights, minvals, maxvals, update_on_read, dtype, data_format, sample_target, label_id, shared_slot):
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

        #get label
        label = f['climate']['labels'][...].astype(np.int32)

    #do min/max normalization
    for c in range(len(channels)):
        data[c,:,:] = (data[c,:,:]-minvals[c])/(maxvals[c]-minvals[c])
    
    #transpose if necessary
    if data_format == "channels_last":
        data = np.transpose(data, [1,2,0])

    #if new dataset is used, label has a batch index.
    #just take the first entry for the moment
    if label.ndim == 3:
        chan = np.random.randint(low=0, high=label.shape[0]) if label_id==None else label_id
        label = label[chan,:,:]

    # cast data and labels if needed: important, do that after min-max norming
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

    def __init__(self, path, channels, weights, dtype, normalization_file=None, update_on_read=False, data_format="channels_first", label_id=None, sample_target=None):
        self.path = path
        self.channels = channels
        self.update_on_read = update_on_read
        self.dtype = dtype.as_numpy_dtype()
        self.weights = np.asarray(weights, dtype=self.dtype)
        self.data_format = data_format
        self.sample_target = sample_target
        self.label_id = label_id
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

    def read(self, datafile, profile=False):
        if isinstance(datafile, bytes):
            datafile = datafile.decode("utf-8")
        path = os.path.join(self.path, datafile)
        if profile: begin_time = time.time()
        #nvtx.RangePush('h5_input', 8)
        shared_slot = smem.get_free_slot()
        data, label, weights, new_minvals, new_maxvals = self.pool.apply(_h5_input_subprocess_reader, (path, self.channels, self.weights, self.minvals, self.maxvals, self.update_on_read, self.dtype, self.data_format, self.sample_target, self.label_id, shared_slot))
        if self.update_on_read:
            self.minvals = np.minimum(self.minvals, new_minvals)
            self.maxvals = np.maximum(self.maxvals, new_maxvals)
        data, label, weights = smem.unpack_arrays(shared_slot, data, label, weights)
        smem.return_slot(shared_slot)
        #nvtx.RangePop()
        if profile: 
            end_time = time.time()
            print("Time to read in parallel %s = %.3f s" % (path, end_time-begin_time))
        return data, label, weights, path

    def sequential_read(self, datafile, profile=False):
        if isinstance(datafile, bytes):
            datafile = datafile.decode("utf-8")
        path = os.path.join(self.path,datafile)
        
        if profile:
            timers = {}
            timers["total"] = -time.time()

        with h5.File(path, "r", driver="core", backing_store=False, libver="latest") as f:
            #get min and max values and update stored values
            if self.update_on_read:
                self.minvals = np.minimum(self.minvals, f['climate']['stats'][self.channels,0])
                self.maxvals = np.maximum(self.maxvals, f['climate']['stats'][self.channels,1])
            #get data
            data = f['climate']['data'][self.channels,:,:]
            if profile: timers["io_data"] = timers["total"] + time.time()

            #get label
            if profile: timers["io_label"] = -time.time()
            label = f['climate']['labels'][...]
            if profile: timers["io_label"] += time.time()
        if profile: timers["io"] = timers["total"] + time.time()
        
        #do min/max normalization
        if profile: timers["norm"] = -time.time()
        for c in range(len(self.channels)):
            data[c,:,:] = (data[c,:,:]-self.minvals[c])/(self.maxvals[c]-self.minvals[c])
        if profile: timers["norm"] += time.time()

        if profile: timers["transpose"] = -time.time()
        if self.data_format == "channels_last":
            data = np.transpose(data, [1,2,0])
        if profile: timers["transpose"] += time.time()

        #if new dataset is used, label has a batch index.
        #just take the first entry for the moment
        if profile: timers["select_channels"] = -time.time()
        if label.ndim == 3:
            chan = np.random.randint(low=0, high=label.shape[0]) if self.label_id==None else self.label_id
            label = label[chan,:,:]
        if profile: timers["select_channels"] += time.time()

        ## cast data and labels if needed
        if profile: timers["cast_data"] = -time.time()
        if data.dtype != self.dtype:
            data = data.astype(self.dtype)
        if profile: timers["cast_data"] += time.time()

        if profile: timers["cast_label"] = -time.time()
        if label.dtype != np.int32:
            label = label.astype(np.int32)
        if profile: timers["cast_label"] += time.time()

        if self.sample_target is not None:
            # determine the number of pixels in each of the three classes
            counts = np.histogram(label, bins=[0,1,2,3])[0]
            # assign a per-class probability that delivers the target number of
            #  pixels of each class (in expectation) in the mask
            prob = float(self.sample_target) / counts
            # randomly select actual pixels to include in the loss function on
            #  this step
            r = np.random.uniform(size=label.shape)
            weights = (r < prob[label]).astype(dtype)
        else:
            #get weights - choose per-channel based on the labels
            weights = self.weights[label]

        #time
        if profile: 
            timers["total"] += time.time()
            for key, val in timers.items():
                print("READ: %s = %.3f s"%(key, val))
            print("")

        return data, label, weights, path


#load data routine
def load_data(input_path, shuffle=True, max_files=-1, use_horovod=True):
    #look for labels and data files
    files = sorted([x for x in os.listdir(input_path) if "data" in x])

    #we will choose to load only the first p files
    if max_files > 0:
        files = files[:max_files]

    #convert to numpy
    files = np.asarray(files, dtype=str)

    #PERMUTATION OF DATA
    if shuffle:
        np.random.seed(12345)
        shuffle_indices = np.random.permutation(len(files))
        files = files[shuffle_indices]

    return files


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
