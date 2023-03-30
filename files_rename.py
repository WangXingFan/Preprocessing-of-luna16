from glob import glob

import numpy as np


path = "E:/VSCode/npy/npy0/"

trainimage = np.load(path + "trainImages.npy")

print(len(trainimage))