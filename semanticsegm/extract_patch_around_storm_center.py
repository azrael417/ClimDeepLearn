from teca import *
import os
import pickle

filename = './wind_tracks_CAM5-1-0.25degree_All-Hist_est1_v3_run2.bin'
#filename = './candidates_CAM5-1-0.25degree_All-Hist_est1_v3_run2.bin'
#filename = './tracks_CAM5-1-0.25degree_All-Hist_est1_v3_run2.bin'

# load the full set of tracks
reader = teca_table_reader.New()
reader.set_file_name(filename)

tap_1 = teca_dataset_capture.New()
tap_1.set_input_connection(reader.get_output_port())
tap_1.update()

table = as_teca_table(tap_1.get_dataset())

# print some info
print 'In Tracks file: %s'%(filename)
print 'Found Columns:'
for i in xrange(table.get_number_of_columns()):
  print '%i - %s'%(i, table.get_column_name(i))
print 'Number of rows: %d'%(table.get_number_of_rows())

# get a specific track
# note that you can use any mathematical/logical experssion here.
# rules are like C++
filt = teca_table_remove_rows.New()
#filt.set_mask_expression('!(track_id==96)')
filt.set_mask_expression('!((year==1996)&&(month==05)&&(day==19))')
filt.set_input_connection(reader.get_output_port())

tap_2 = teca_dataset_capture.New()
tap_2.set_input_connection(filt.get_output_port())
tap_2.update()

track_96 = as_teca_table(tap_2.get_dataset())

# get the positions of points on the track
# get radial size of the storm
track_id = track_96.get_column('track_id').as_array()
lon_96 = track_96.get_column('lon').as_array()
lat_96 = track_96.get_column('lat').as_array()
r0_96 = track_96.get_column('wind_radius_0').as_array()
wind_96 = track_96.get_column('surface_wind').as_array()
time_96 = track_96.get_column('time').as_array()
year_96 = track_96.get_column('year').as_array()
month_96 = track_96.get_column('month').as_array()
day_96 = track_96.get_column('day').as_array()
hour_96 = track_96.get_column('hour').as_array()
minute_96 = track_96.get_column('minute').as_array()
print 'Track 96'
print 'Number of rows in track 96: %d'%(track_96.get_number_of_rows())
print 'lon lat size wind time year month day hour minute'
for i in xrange(len(lon_96)):
  print '%f %f %f %f %f %f %f %f %f %f %f'%(track_id[i],lon_96[i], lat_96[i], r0_96[i], wind_96[i], time_96[i], year_96[i], month_96[i], day_96[i], hour_96[i], minute_96[i])

data = {}
lat = []
lon = []
r_0 = []
wind = []
print("Number of entries from this extraction is {}".format(year_96.shape[0]))
for ii in range(year_96.shape[0]):
   #if ii > 0 and (day_96[ii] != day_96[ii-1]):
   fname = "CAM5-1-0.25degree_All-Hist_est1_v3_run2.cam.h2."+str(year_96[ii])+"-"+str(month_96[ii]).zfill(2)+"-"+str(day_96[ii]).zfill(2)+"-00000.pkl"
   data['track_id'] = track_id
   data['year'] = year_96
   data['month'] = month_96
   data['day'] = day_96
   data['hour'] = hour_96
   data['minute'] = minute_96
   data['lon'] = lon
   data['lat'] = lat
   data['r_0'] = r_0 
   data['wind'] = wind 
   pickle.dump(data,open(fname,'wb'))
   lat.append(lat_96[ii])
   lon.append(lon_96[ii])
   r_0.append(r0_96[ii])
   wind.append(wind_96[ii])
   #h2filename = "/global/cscratch1/sd/mwehner/CAM5-1-0.25degree_All-Hist_est1_v3_run2/run/h2/CAM5-1-0.25degree_All-Hist_est1_v3_run2.cam.h2."+str(year_96[ii])+"-"+str(month_96[ii]).zfill(2)+"-"+str(day_96[ii]).zfill(2)+"-00000.nc"
   #NEED TO VERIFY UNIQUENESS of year, month, day. If unique  write out old dict to pkl , create new dict
   #If not unique append
   #print(h2filename)
