import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import copy
import keras


imgs = np.load("/home/mudigonda/Data/multi_channel_images.npy")
imgs = imgs.reshape([imgs.shape[0],imgs.shape[1],imgs.shape[2],3])
labels = np.load("/home/mudigonda/Data/tiramisu_clipped_combined_v2/masks.npy")

print("loaded data)")

#Image metadata contains year, month, day, time_step, and lat/ lon data for each crop
#See README in $SCRATCH/segmentation_labels/dump_v4 on CORI
image_metadata = np.load("/home/mudigonda/Data/tiramisu_clipped_combined_v2/image_metadata.npy")

imgs = imgs[:,3:-3,...]
labels = labels[:,3:-3,:]

#PERMUTATION OF DATA
np.random.seed(12345)
shuffle_indices = np.random.permutation(len(imgs))
np.save("./shuffle_indices.npy", shuffle_indices)
imgs = imgs[shuffle_indices]
labels = labels[shuffle_indices]
image_metadata = image_metadata[shuffle_indices]

#Create train/validation/test split
trn = imgs[:int(0.8*len(imgs))]
trn_labels = labels[:int(0.8 * len(labels))]
test = imgs[int(0.8*len(imgs)):int(0.9*len(imgs))]
test_labels = labels[int(0.8*len(imgs)):int(0.9*len(imgs))]
valid = imgs[int(0.9*len(imgs)):]
valid_labels = labels[int(0.9*len(imgs)):]

num_test_images = test.shape[0]

#load model
model = keras.models.load_model("/home/mudigonda/Data/multi_channel_training_round3/weights-13-2.60-5-0770-.hdf5")
print("loaded model")

for i, test_rank in enumerate(test):
        #Plot the model output
	    xx,yy = np.meshgrid(np.arange(144), np.arange(96))
	    cmap = copy.copy(plt.cm.get_cmap('bwr'))       
	    cmap.set_bad(alpha=0)
	    #Replace the 2 with the index of the channel you want to plot
        test_img_plt = test_img[:, :, 2]
	    plt.contourf(xx,yy,test_img_plt.squeeze(),cmap='viridis',alpha=0.8)
	    cbar = plt.colorbar()

	    #Plot the gt labels
        plt.contourf(xx,yy,np.ma.masked_less(test_labels.squeeze(), 0.5),cmap=cmap,vmin=1,vmax=2,alpha=0.7)
	    plt.title("Ground Truth Labels of ARs and TCs")
	    cbar.ax.set_ylabel('TMQ kg $m^{-2}$')
	    plt.gcf().savefig("./images_after_first_multi/gt"+str(total)+".png")
	    plt.clf()
	    
	    #Run the model on the current test image                                                                               
	    pred = model.predict(test_img[np.newaxis,:, :, :])                                                          
	    p = np.argmax(pred[0],-1).reshape(image_height,image_width)
	    p = np.ma.masked_less(p,0.5)
	    #Plot the same channel as above                                                            
	    plt.contourf(xx,yy,test_img_plt.squeeze(),cmap='viridis',alpha=0.8)
	    cbar = plt.colorbar()
	    #Plot the prediction                             
	    plt.contourf(xx, yy, p, cmap=cmap,alpha=0.7,vmin=1,vmax=2)
	    plt.title("5-Layer DenseNet Prediction")
	    cbar.ax.set_ylabel('TMQ kg $m^{-2}$')
	    plt.gcf().savefig("./images_after_first_multi/pred" + str(total)+".png")
	    plt.clf()
        total += 1


preds = model.predict(test, verbose=1)   
p = np.argmax(preds,-1).reshape(num_test_images, image_height,image_width)   
cm = sklearn.metrics.confusion_matrix(test_labels.reshape(num_test_images*96*144),p.reshape(num_test_images*96*144) )    
import IPython; IPython.embed()
