#!/usr/bin/env python

import numpy as np
from scipy.misc import imsave
import sys
import glob
import re

#                                       label predict color
colormap = np.array([[[  0,  0,  0],  #   0      0     black
                      [255,  0,255],  #   0      1     purple
                      [  0,255,255]], #   0      2     cyan
                     [[  0,255,  0],  #   1      0     green
                      [128,128,128],  #   1      1     grey
                      [255,255,  0]], #   1      2     yellow
                     [[255,  0,  0],  #   2      0     red
                      [  0,  0,255],  #   2      1     blue
                      [255,255,255]], #   2      2     white
                     ])

for pred in glob.glob(sys.argv[1]):

    outfilename = re.sub('\.npz', '.gif', pred)

    data = np.load(sys.argv[1])
    pdata = data["prediction"] / 100
    ldata = data["label"] / 100

    outdata = colormap[ldata.astype(int),pdata.astype(int)]
    imsave(outfilename, outdata)
    
