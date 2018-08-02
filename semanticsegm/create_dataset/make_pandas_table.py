import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join, basename
import argparse
from glob import glob

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--numpy_dir", type=str, help='directory of numpy files generated from TECA bin')
	parser.add_argument('--output_dir', type=str, help='where to write the pandas version of the TECA binary file')
	cli_args = parser.parse_args()

numpy_files = glob(cli_args.numpy_dir + "*.npy")

#filenames.remove("make_pandas_table.py")
#filenames.remove("make_pandas_table.py~")

data = {}

for f in numpy_files:
	var_name = basename(f).replace('.npy', '')
	data[var_name] = np.load(f)

df = pd.DataFrame(data)
print(data.keys())

df.to_csv("{}teca_labels.csv".format(cli_args.output_dir), index=False)
