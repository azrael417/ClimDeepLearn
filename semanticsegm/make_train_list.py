import glob
import random
import pickle
import numpy as np

if __name__ == "__main__":
  files = glob.glob('/global/cscratch1/sd/tkurth/gb2018/tiramisu/segm_h5_v3_reformat/*')
  random.shuffle(files)
  TRAIN = 0.8 
  VAL = 0.1
  TEST = 1 - TRAIN - VAL
  TOT = len(files)
  print("indices {}".format(int(np.floor(TOT*TRAIN))))
  print("val_indinces {}".format(int(np.floor(TOT*(TRAIN+VAL)))))
  train_files = files[0:int(np.floor(TOT*TRAIN))]
  val_files = files[int(np.floor(TOT*TRAIN)): int(np.floor(TOT*(TRAIN+VAL)))]
  test_files = files[int(np.floor(TOT*(TRAIN+VAL))):]
  file_list = {}
  file_list['train'] = train_files
  file_list['validation'] = val_files
  file_list['test'] = test_files
  with open("shuffle_fname.txt","wb") as fp:
     pickle.dump(file_list,fp)

  
  
