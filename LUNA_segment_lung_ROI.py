from matplotlib import pyplot as plt
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from glob import glob
import numpy as np

working_path = "E:\VSCode\lung/npy/"
file_list = glob(working_path + "images_*.npy")

for img_file in file_list:
    imgs_to_process = np.load(img_file).astype(np.float64) 
    print("on image",img_file)
    for i in range(len(imgs_to_process)):
        img = imgs_to_process[i]

        #标准化
        mean = np.mean(img)   #均值
        std = np.std(img)    #标准差
        img = (img-mean)/std

        #寻找肺部附近的平均像素,以重新调整过度曝光的图像
        middle = img[100:400,100:400]
        mean = np.mean(middle)
        max = np.max(img)
        min = np.min(img)

        img[img==max] = mean
        img[img==min] = mean

        #使用Kmeans算法将前景（放射性不透明组织）和背景（放射性透明组织，即肺部）分离。
        #仅在图像中心进行此操作，以尽可能避免图像的非组织部分。
        kmeans = KMeans(n_clusters=2,n_init=10).fit(np.reshape(middle,[np.prod(middle.shape),1]))
        #np.prod 计算给定数组中所有元素的乘积
        #np.reshape 低维变高维  .flatten() 高维变1维
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)
        thresh_img = np.where(img<threshold,1.0,0.0)  #阈值化图像，二值化处理

        #腐蚀和膨胀
        eroded = morphology.erosion(thresh_img,np.ones([4,4]))
        dilation = morphology.dilation(eroded,np.ones([10,10]))

        labels = measure.label(dilation)#对二值图像进行标记，标记连通区域
        label_vals = np.unique(labels)  #获取标记值的唯一值，即标记的数量
        regions = measure.regionprops(labels) #标记区域
        good_labels = []
        for prop in regions:
            B = prop.bbox   #边界框
            if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
                good_labels.append(prop.label)
            #np.ndarray 创建多维数组
        mask = np.ndarray([512,512],dtype=np.int8)
        mask[:] = 0  #肺部mask

        for N in good_labels:
            mask = mask + np.where(labels==N,1,0)
        mask = morphology.dilation(mask,np.ones([10,10]))
        imgs_to_process[i] = mask

    np.save(img_file.replace("images","lungmask"),imgs_to_process)





