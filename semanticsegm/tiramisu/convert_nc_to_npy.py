import numpy as np
import netCDF4 as nc
import glob

dataset_dir = '/home/mudigonda/files_for_first_maskrcnn_test/'
data_filenames = glob.glob(dataset_dir + '[1-2]*boxes.npy')
save_dir = '/home/mudigonda/Data/tiramisu_images/'

def _process_img_id_string(img_id_string):
  year = int(img_id_string[:4])
  month = int(img_id_string[4:6])
  day = int(img_id_string[6:8])
  time_step = int(img_id_string[8:10])
  return year, month, day, time_step

def load_image(filepath, year, month, day, time_step):
    filepath += "CAM5-1-0.25degree_All-Hist_est1_v3_run2.cam.h2." +"{:04d}-{:02d}-{:02d}-00000.nc".format(year, month, day)
    print(filepath)
    with nc.Dataset(filepath) as fin:
        TMQ = fin['TMQ'][:][time_step][:,:,np.newaxis]
        print(TMQ.shape)
    return (TMQ * 1000).astype('uint32')


for data_filename in data_filenames:
  id_start_index = min([i for i, c in enumerate(data_filename) if c.isdigit()])
  img_id_string = data_filename[id_start_index:id_start_index+10]
  img_id = int(img_id_string)

  year, month, day, time_step = _process_img_id_string(img_id_string)

  TMQ = load_image(dataset_dir, year, month, day, time_step)
  
  np.save(save_dir+img_id_string+"_image.npy",TMQ)