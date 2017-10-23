from teca import *
import os
import pickle
import csv
#import pandas as pd

filename = './wind_tracks_CAM5-1-0.25degree_All-Hist_est1_v3_run2.bin'
#filename = './candidates_CAM5-1-0.25degree_All-Hist_est1_v3_run2.bin'
#filename = './tables_CAM5-1-0.25degree_All-Hist_est1_v3_run2.bin'

# load the full set of tables
reader = teca_table_reader.New()
reader.set_file_name(filename)

tap_1 = teca_dataset_capture.New()
tap_1.set_input_connection(reader.get_output_port())
tap_1.update()

table = as_teca_table(tap_1.get_dataset())

# print some info
print 'In tables file: %s'%(filename)
print 'Found Columns:'
for i in xrange(table.get_number_of_columns()):
  print '%i - %s'%(i, table.get_column_name(i))
print 'Number of rows: %d'%(table.get_number_of_rows())

track_id = table.get_column('track_id').as_array()
lon = table.get_column('lon').as_array()
lat = table.get_column('lat').as_array()
r0 = table.get_column('wind_radius_0').as_array()
wind = table.get_column('surface_wind').as_array()
time = table.get_column('time').as_array()
year = table.get_column('year').as_array()
month = table.get_column('month').as_array()
day = table.get_column('day').as_array()
hour = table.get_column('hour').as_array()
storm_id = table.get_column('storm_id').as_array()

print(track_id.size)
print(lon.size)
print(lat.size)
print(r0.size)
print(wind.size)
print(time.size)
print(year.size)
print(month.size)
print(day.size)
print(hour.size)
print(storm_id.size)

save_filepath = "/global/cscratch1/sd/amahesh/segmentation_labels"

np.save(save_filepath+"track_id", track_id)
np.save(save_filepath+"lon", lon)
np.save(save_filepath+"lat", lat)
np.save(save_filepath+"r0", r0)
np.save(save_filepath+"wind", wind)
np.save(save_filepath+"time", time)
np.save(save_filepath+"year", year)
np.save(save_filepath+"month", month)
np.save(save_filepath+"day", day)
np.save(save_filepath+"hour", hour)
np.save(save_filepath+"storm_id", storm_id)
