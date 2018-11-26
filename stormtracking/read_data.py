from random import shuffle
import numpy as np
import copy

fname="sort_by_time.csv"
ts=24;
#Seperate as tensor(in and out)
with open(fname) as f:
    content = f.readlines();
    time_list=[];lat_list=[]; lon_list=[]; num_list=[]; traj_list=[];
    lon_day=[];lat_day=[];
    time_list.append(content[0].split(",")[0])
    for i in range(len(content)):
        #input
        line=content[i].split(",")
        time=line[0];lon=line[1];lat=line[2];track_id=line[3];
        if (i==0) or (time==content[i-1].split(",")[0]):
            lon_day.append(lon); lat_day.append(lat);
            lon_day.append(track_id); lat_day.append(track_id);
        elif (i==len(content)-1):
            print(lon_day,lat_day)
            lon_list.append(lon_day); lat_list.append(lat_day);
            lon_day=[]; lat_day=[];
            lon_day.append(lon); lat_day.append(lat);
            lon_day.append(track_id); lat_day.append(track_id);
            lon_list.append(lon_day); lat_list.append(lat_day);
            time_list.append(time);
        else:
            lon_list.append(lon_day); lat_list.append(lat_day);
            time_list.append(time);
            lon_day=[]; lat_day=[];
            lon_day.append(lon); lat_day.append(lat);
            lon_day.append(track_id); lat_day.append(track_id);



def make_file_input(kk,bs): 
    #*********************INPUT******************************#
    input_data=[];
    file_output_index=[];
    overhead=47 #(ts)+ts-1;
    for i in xrange(kk,kk+overhead): #len(lon_list)):#is_47615
        output=[0]*60*30;
        output_i=[];
        for j in range(len(lon_list[i])/2):
            xx=int(float(lon_list[i][2*j])/6); yy=int((180-(int(float(lat_list[i][2*j]))+90))/6);
            index=yy*60+xx; #Image is 60 x 30
            output[index]=1;
            output_i.append(index);
        input_data.append(output);
        file_output_index.append(output_i);
    return input_data

def make_file_output(kk,batch_size):
    #*********************OUTPUT***************************#
    output_state=[]; output_lonlat=[]; 
    for i in xrange(kk,kk+batch_size): #len(lon_list)):#is_47615
        output_state_week=[[0]*13 for ii in range(ts)]; output_lonlat_week=[[[0,0]]*13 for ii in range(ts)]; traj_id_week=[]; # LonLat: rescaled as btw 0 to 1, -1 is no     
        for j in range(ts): # j is counter of day in 3 days
            day_index=i+j;
            lon_temp=[]; lat_temp=[]; traj_temp=[];
            # Sort according to ascending order of lon variable
            for k in range(len(lon_list[day_index])/2):
                lon_temp.append((float(lon_list[day_index][2*k])+180.0)/360.0); #RESCALED
                lat_temp.append((float(lat_list[day_index][2*k])+90.0)/180.0);  #RESCALED
                traj_temp.append(lon_list[day_index][2*k+1]);
            sorted_lon=sorted(lon_temp); sorted_lat=[]; sorted_traj=[];
            for k in range(len(sorted_lon)):
                sorted_lat.append(lat_temp[lon_temp.index(sorted_lon[k])]);
                sorted_traj.append(traj_temp[lon_temp.index(sorted_lon[k])]);
            #CHECK Traj exist in previous days
            for k in range(len(sorted_traj)):
               #CHECK Whether its old traj
                Is_it_new=1;
                for ll in range(len(traj_id_week)):
                    if (sorted_traj[k]==traj_id_week[ll]):
                        output_state_week[j][ll]=1;
                        output_lonlat_week[j][ll]=[sorted_lon[k],sorted_lat[k]];
                        Is_it_new=0;
                #If its new
                if (Is_it_new==1):
                    ll=len(traj_id_week);
                    traj_id_week.append(sorted_traj[k]);
                    output_state_week[j][ll]=1;
                    output_lonlat_week[j][ll]=[sorted_lon[k],sorted_lat[k]];
        output_state.append(output_state_week);
        output_lonlat.append(output_lonlat_week);
    return output_state, output_lonlat

def read_input(batch_size,train_size, test_size):
    input_batch=[]; output_state_batch=[]; output_lonlat_batch=[];
    file_lenth=len(lon_list);
    for kk in range(test_size+train_size): #fiile_lenth-115):
        output_state, output_lonlat = make_file_output(kk,batch_size);
        input_image=[]; 
        for i in range(batch_size):
            temp_list  = make_file_input(kk,batch_size);
            for j in xrange(1,ts-i):
               back_del=-1*(j);
               del temp_list[back_del];
            for k in xrange(0,i):
               front_del=k;
               del temp_list[front_del];
            input_image.append(temp_list)
        if(kk%100==0): print(str(kk)+"th data read \n");
        input_batch.append(input_image); 
        output_state_batch.append(output_state); 
        output_lonlat_batch.append(output_lonlat);
    index=[0+i for i in range(len(output_state_batch))];
#    index=[0,1,2,3,4,5,6,7,8,9];
    print(len(output_state)); 
    shuffle(index)
    tr_input_batch=[]; tr_output_state_batch=[]; tr_output_lonlat_batch=[];
    te_input_batch=[]; te_output_state_batch=[]; te_output_lonlat_batch=[];
    for i in range(train_size):
        tr_input_batch.append(input_batch[index[i]]);
        tr_output_state_batch.append(output_state_batch[i]);
        tr_output_lonlat_batch.append(output_lonlat_batch[i]);
    for j in xrange(train_size,train_size+test_size):
        te_input_batch.append(input_batch[index[j]]);
        te_output_state_batch.append(output_state_batch[j]);
        te_output_lonlat_batch.append(output_lonlat_batch[j]);
    return tr_input_batch, tr_output_state_batch, tr_output_lonlat_batch, te_input_batch, te_output_state_batch, te_output_lonlat_batch;
##inn,o1,o2=read_input();  
#tr_input_batch, tr_output_state_batch, tr_output_lonlat_batch, te_input_batch, te_output_state_batch, te_output_lonlat_batch=read_input(24,8,2);
#print np.shape(tr_input_batch);
#print np.shape(tr_output_lonlat_batch);
#print np.shape(tr_output_state_batch);
#print np.shape(te_input_batch);
#print np.shape(te_output_lonlat_batch);
#print np.shape(te_output_state_batch); 
