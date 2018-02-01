import numpy as np
import keras

#Load the images and the labels
imgs = np.load("/home/mudigonda/Data/tiramisu_clipped_combined_v2/images.npy")
imgs = imgs.reshape([imgs.shape[0],imgs.shape[1],imgs.shape[2],1])
labels = np.load("/home/mudigonda/Data/tiramisu_clipped_combined_v2/masks.npy")

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

#load model
model = keras.models.load_model("path/to/model")

for i,test_img in enumerate(test):
    xx,yy = np.meshgrid(np.arange(144), np.arange(96))
    plt.contourf(xx,yy,test_img.squeeze(),cmap='viridis')
    cbar = plt.colorbar()
    plt.contourf(xx,yy,test_labels[i],cmap='gray',alpha=0.45)
    plt.title("Ground Truth Labels of ARs and TCs")
    cbar.ax.set_ylabel('TMQ kg $m^{-2}$')
    plt.gcf().savefig("./images_after_first_model/gt"+str(i)+".png")
    plt.clf()

    pred = model.predict(test[i:i+1])
    p = np.argmax(pred[0],-1).reshape(image_height,image_width)
    plt.contourf(xx,yy,test_img.squeeze(),cmap='viridis')
    cbar = plt.colorbar()
    plt.contourf(xx, yy, p, cmap='gray',alpha=0.45)
    plt.title("5-Layer DenseNet Prediction")
    cbar.ax.set_ylabel('TMQ kg $m^{-2}$')
    plt.gcf().savefig("./images_after_first_model/pred" + str(i)+".png")
    plt.clf()
