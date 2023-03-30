from glob import glob
import os

import numpy as np


root_path = "E:/VSCode/npy/"

images_list = []
lungmasks_list = []
masks_list = []

for i in range(10):
    npy_path = root_path + "npy" + str(i) + "/"
    img_files = glob(npy_path + "images_*.npy")
    lung_files = glob(npy_path + "lungmask_*.npy")
    mask_list = glob(npy_path + "masks_*.npy")

    for f in img_files:
        images_list.append(f)
    for f in lung_files:
        lungmasks_list.append(f)
    for f in mask_list:
        masks_list.append(f)

print(len(images_list),len(lungmasks_list),len(masks_list))

img = images_list[100]
img = os.path.basename(img)
print(img)
