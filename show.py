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

fname = lungmask_list[5]
imgs = img_list[5]

i=0
masks = np.load(fname)
imgs = np.load(imgs)
mask = masks[i]
label = measure.label(mask)
regions = measure.regionprops(label)
for prop in regions:
    B = prop.bbox
    print(B)

# imgs = np.load(img_list[20])
# mask = np.load(mask_list[20])
# lungmask = np.load(lungmask_list[20])

# for i in range(len(imgs)):
#     print("img %d" %i)
#     fig,ax = plt.subplots(2,2,figsize=[8,8])
#     ax[0,0].imshow(imgs[i],cmap = "gray")
#     ax[0,1].imshow(mask[i],cmap="gray")
#     ax[1,0].imshow(imgs[i]*mask[i],cmap="gray")
#     ax[1,1].imshow(lungmask[i],cmap = "gray")
#     plt.show()