from glob import glob
import os

import numpy as np


root_path = "E:/VSCode/npy/"
new_path = "E:/VSCode/npy/npy/"

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
for i in range(len(lungmasks_list)):
    old_name = lungmasks_list[i]
    if (i < 9):
        new_name = new_path + "lungmask_000" + str(i+1) + ".npy"
    elif(i < 99):
        new_name = new_path + "lungmask_00" + str(i+1) + ".npy"
    elif(i < 999):
        new_name = new_path + "lungmask_0" + str(i+1) + ".npy"
    else:
        new_name = new_path + "lungmask_" + str(i+1) + ".npy"
    os.rename(old_name,new_name)

    

