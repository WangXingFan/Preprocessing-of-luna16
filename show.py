from glob import glob
#import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure

path = "npy/"

img_list = glob(path + "images_*.npy")
mask_list = glob(path + "masks_*.npy")
#lungmask_list = glob("E:\LUNA16\TianChi_Rank22_npy/npy/" + "lungmask_*.npy")
lungmask_list = glob(path + "lungmask_*.npy")
#print(len(img_list),len(mask_list)


trainImage = np.load(path + "trainImages.npy")
trainMask = np.load(path + "trainMasks.npy")
img = trainImage[5]
mask = trainMask[5]

plt.subplot(1,2,1)
plt.imshow(img[0],cmap = "gray")
plt.subplot(1,2,2)
plt.imshow(mask[0],cmap = "gray")
plt.show()