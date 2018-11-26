import tensorflow as tf
import h5py as h5
import argparse
import glob
import os
import numpy as np

def write_tf_records(data,labels,stats,tfrec_fName):

    def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    writer = tf.python_io.TFRecordWriter(tfrec_fName)
    channels, height, width = data.shape
    data_raw = data.tobytes()
    labels_raw = labels.tobytes()
    stats_raw = stats.tobytes()
    example = tf.train.Example(features = tf.train.Features(feature={
	    'channel': _int64_feature(channels),
	    'height': _int64_feature(height),
	    'width': _int64_feature(width),
	    'data' : _bytes_feature(data_raw),
	    'labels': _bytes_feature(labels_raw),
	    'stats': _bytes_feature(stats_raw)}))
    writer.write(example.SerializeToString())
    writer.close()
    return None

def read_h5(datafile):
    with h5.File(datafile, "r", driver="core", backing_store=False, libver="latest") as f:
         #get min and max values and update stored values
         stats = f['climate']['stats'][:]
         #get data
         data = f['climate']['data'][:].astype(np.float32)

         #get label
         label = f['climate']['labels'][:].astype(np.int32)
     #time
     #end_time = time.time()
     #print "Time to read image %.3f s" % (end_time-begin_time)
    return data, label, stats 

def main():
   AP = argparse.ArgumentParser()
   AP.add_argument("--input",type=str,default=None)
   AP.add_argument("--output",type=str,default=None)
   parsed = AP.parse_args()
   files = glob.glob(parsed.input+"/*")
   #create output path if not exists
   if not os.path.isdir(parsed.output):
      os.makedirs(parsed.output)
   
   for ii in range(2):
	print("File name that is being read is {} ".format(files[ii]))
  	data,label, stats = read_h5(files[ii])
  	write_tf_records(data,label,stats,parsed.output+files[ii].split('/')[-1])

if __name__ == "__main__":
	main()
