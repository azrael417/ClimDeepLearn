import pandas as pd 
import numpy as np
from calendar import monthrange

teca_table = pd.read_csv("/global/cscratch1/sd/amahesh/segmentation_labels/teca_labels_HAPPI20.csv")

for year in range(2106,2115):
	for month in range(1,13):
		for day in range(1,monthrange(year, month)[1]+1):
                        for run_num in [1,2,3,4,6]:
                                curr_table = teca_table[np.logical_and(np.logical_and(np.logical_and((teca_table['day'] == day), \
				                                                                     (teca_table['month'] == month)),teca_table['year'] == year), teca_table['run_nums'] == run_num)]
                                if len(curr_table) > 0:
                                        curr_table.to_csv("/global/cscratch1/sd/amahesh/segmentation_labels/teca_subtables_HAPPI20/teca_labels_{:04d}-{:02d}-{:02d}-{:01d}.csv".format(year,month,day, run_num),index=False)
                                else:
                                        print([year,month,day,run_num])
