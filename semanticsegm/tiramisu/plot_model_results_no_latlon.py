import numpy as np

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