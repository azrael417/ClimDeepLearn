import os
import numpy as np
import argparse as ap
import pandas as pd
import datetime as dt
import dateutil.parser as dp
from shutil import copyfile

def main():
    parser = ap.ArgumentParser()
    parser.add_argument('--srcpath', type=str, required=True,
                        help='path to source files')
    parser.add_argument('--dstpath_train', type=str, required=True,
                        help='target directory for training files')
    parser.add_argument('--dstpath_test', type=str, required=True,
                        help='target directory for testing files')
    parser.add_argument('--dstpath_validation', type=str, required=True,
                        help='target directory for validation files')
    parser.add_argument('--split', type=float, nargs=3, default=[0.8,0.1,0.1],
                help='split fractions for train,validation,test')
    parser.add_argument('--prefix', type=str, default='',
                        help='optional prefix to mark the data files')
    parser.add_argument('--block_size', type=int, default=1,
                        help='number of consecutive images grabbed')
    parser.add_argument('--stride_size', type=int, default=1,
                        help='slide block window by stride_size entries')
    parser.add_argument('--copy_files', action='store_true',
                        help='copy filesinstead of creating symbolic links when specified')
    args = parser.parse_args()

    if args.block_size!=args.stride_size:
        print("Warning: if stride_size!=block_size, you will have a small overlap between train and validation and validation and test!")

    #do sanity check
    split = args.split
    if np.sum(split) != 1.:
        raise ValueError("Error: please make sure that the splits sum to 1.")

    #check if input directory is actually there
    if not os.path.exists(args.srcpath):
        raise ValueError("Error: the path {} does not exist".format(args.srcpath))

    #get list of files
    filelist = [x for x in os.listdir(args.srcpath) if x.startswith(args.prefix)]

    #convert to records
    reclist = [{'date': dp.parse("-".join(x.split("-")[1:-2])), 'time': int(x.split("-")[-2]), 'simulation': int(x.replace('.h5','').split("-")[-1]), 'filename': x} for x in filelist]

    #convert to dataframe
    datadf = pd.DataFrame(reclist)

    #sort the frame:
    datadf.sort_values(by=['simulation','date', 'time'], inplace=True)
    datadf.reset_index(inplace=True, drop=True)
    
    #do the block selection
    selectdf = datadf.groupby('simulation').apply(lambda x: [x['filename'].iloc[i:i+args.block_size] for i in range(0,len(x['filename'])-args.block_size,args.stride_size)])

    #join over simulations
    combinedlist=[]
    for sublist in list(selectdf.values.flatten()):
        combinedlist+=sublist

    #do split in train, test, validation: NOTE THAT IF stride_size!=block-size you will have a small overlap between train and val and val and test
    #shuffle
    np.random.seed(12345)
    perm = np.random.permutation(len(combinedlist))
    shuffledlist = [combinedlist[p] for p in perm]

    #split
    num_elem = len(shuffledlist)
    trainfiles = shuffledlist[:int(num_elem*split[0])]
    validationfiles = shuffledlist[int(num_elem*split[0]):int(num_elem*split[0])+int(num_elem*split[1])]
    testfiles = shuffledlist[int(num_elem*split[0])+int(num_elem*split[1]):]
    
    #make split manifest:
    #train
    #create directory structure if not existing
    if not os.path.exists(args.dstpath_train):
        os.makedirs(args.dstpath_train)
    for traintuples in trainfiles:
        for trainfile in traintuples:
            srcfile=os.path.join(args.srcpath,trainfile)
            dstfile=os.path.join(args.dstpath_train,trainfile)
            if not os.path.exists(dstfile):
                if args.copy_files:
                    copyfile(srcfile, dstfile)
                else:
                    os.symlink(srcfile, dstfile)

    #validation
    #create directory structure if not existing
    if not os.path.exists(args.dstpath_validation):
        os.makedirs(args.dstpath_validation)
    for validationtuples in validationfiles:
        for validationfile in validationtuples:
            srcfile=os.path.join(args.srcpath,validationfile)
            dstfile=os.path.join(args.dstpath_validation,validationfile)
            if not os.path.exists(dstfile):
                if args.copy_files:
                    copyfile(srcfile, dstfile)
                else:
                    os.symlink(srcfile, dstfile)

    #test
    #create directory structure if not existing
    if not os.path.exists(args.dstpath_test):
        os.makedirs(args.dstpath_test)
    for testtuples in testfiles:
        for testfile in testtuples:
            srcfile=os.path.join(args.srcpath,testfile)
            dstfile=os.path.join(args.dstpath_test,testfile)
            if not os.path.exists(dstfile):
                if args.copy_files:
                    copyfile(srcfile, dstfile)
                else:
                    os.symlink(srcfile, dstfile)


if __name__ == '__main__':
    main()
