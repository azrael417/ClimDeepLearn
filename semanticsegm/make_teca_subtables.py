import pandas as pd 
import numpy as np
from calendar import monthrange
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--table_path", type=str, help='path to table')
	parser.add_argument('--output_dir', type=str, help='where to write the pandas version of the TECA binary file')
	cli_args = parser.parse_args()


teca_table = pd.read_csv(cli_args.table_path)
print(np.min(teca_table['year']),np.max(teca_table['year']))


for year in range(np.min(teca_table['year']),np.max(teca_table['year'])+1):
	for month in range(1,13):
		for day in range(1,monthrange(year, month)[1]+1):
                        for run_num in np.unique(teca_table['run_num']):
                                curr_table = teca_table[np.logical_and(np.logical_and(np.logical_and((teca_table['day'] == day), \
				                                                                     (teca_table['month'] == month)),teca_table['year'] == year), teca_table['run_num'] == run_num)]
                                # if len(curr_table) > 0:
                                curr_table.to_csv("{}teca_labels_{:04d}-{:02d}-{:02d}-{:01d}.csv".format(cli_args.output_dir, year,month,day, run_num),index=False)
                                # else:
                                # print([year,month,day,run_num])