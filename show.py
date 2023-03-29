from glob import glob
#import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np

path = "npy/"

img_list = glob(path + "images_*.npy")
mask_list = glob(path + "masks_*.npy")
#print(len(img_list),len(mask_list))

imgs = np.load(img_list[40])
mask = np.load(mask_list[40])

for i in range(len(imgs)):
    print("img %d" %i)
    fig,ax = plt.subplots(2,2,figsize=[8,8])
    ax[0,0].imshow(imgs[i],cmap = "gray")
    ax[0,1].imshow(mask[i],cmap="gray")
    ax[1,0].imshow(imgs[i]*mask[i],cmap="gray")
    plt.show()