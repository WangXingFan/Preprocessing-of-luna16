from glob import glob
from skimage import measure
from skimage.transform import resize
import numpy as np

working_path = "E:/VSCode/npy/npy0/"

lungmask_list = glob(working_path + "lungmask_*.npy")
out_images = []
out_nodemasks = []
for fname in lungmask_list:
    print("working on file ",fname)
    imgs_to_process = np.load(fname.replace("lungmask","images"))
    masks = np.load(fname)
    node_masks = np.load(fname.replace("lungmask","masks"))
    for i in range(len(imgs_to_process)):
        mask = masks[i]
        node_mask = node_masks[i]
        img = imgs_to_process[i]
        new_size = [512,512]
        img = mask*img

        #ROI区域归一化
        new_mean = np.mean(img[mask>0])
        new_std  = np.std(img[mask>0])
        old_min = np.min(img)
        img[img == old_min] = new_mean-1.2*new_std
        img = (img-new_mean)/(new_std)

        #创建图像的边界框
        labels = measure.label(mask)
        regions = measure.regionprops(labels)

        #找所有区域中的最值边界
        min_row = 512
        max_row = 0
        min_col = 512
        max_col = 0
        for prop in regions:
            B = prop.bbox
            if min_row > B[0]:
                min_row = B[0]
            if min_col > B[1]:
                min_col = B[1]
            if max_row < B[2]:
                max_row = B[2]
            if max_col < B[3]:
                max_col = B[3]
        height = max_row-min_row
        width = max_col-min_col
        if width > height:
            max_row = min_row + width
        else:
            max_col = min_col + height

        img = img[min_row:max_row,min_col:max_col]
        mask = mask[min_row:max_row,min_col:max_col]
#        print("num_row: ",max_row-min_row)
#        print("num_col: ",max_col-min_col)
        if max_row-min_row < 5 or max_col-min_col < 5:
            pass
        else:
            print(2)
            mean = np.mean(img)
            max = np.max(img)
            min = np.min(img)
            img = (img-mean)/(max-min)
            new_img = resize(img,[512,512])
            new_node_mask = resize(node_mask[min_row:max_row,min_col:max_col],[512,512])
            out_images.append(new_img)
#            print(len(out_images))
            out_nodemasks.append(new_node_mask)


num_images = len(out_images)

#将图像和掩膜变成单通道，便于训练
final_images = np.ndarray([num_images,1,512,512],dtype=np.float32)
final_masks = np.ndarray([num_images,1,512,512],dtype=np.float32)
for i in range(num_images):
    final_images[i,0] = out_images[i]
    final_masks[i,0] = out_nodemasks[i]

rand_i = np.random.choice(range(num_images),size=num_images,replace=False).astype(int)
test_i = int(0.2*num_images) 

np.save(working_path+"trainImages.npy",final_images[rand_i[test_i:]])
np.save(working_path+"trainMasks.npy",final_masks[rand_i[test_i:]])
np.save(working_path+"testImages.npy",final_images[rand_i[:test_i]])
np.save(working_path+"testMasks.npy",final_masks[rand_i[:test_i]])
    