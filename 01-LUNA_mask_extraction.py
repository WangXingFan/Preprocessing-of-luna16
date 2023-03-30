import os
import SimpleITK as sitk
import numpy as np
from glob import glob
import pandas as pd
try:
    from tqdm import tqdm # long waits are not fun
except:
    print('TQDM does make much nicer wait bars...')
    tqdm = lambda x: x

def normalizePlanes(npzarray):
    maxHU = 400
    minHU = -1000
    npzarray = (npzarray - minHU)/(maxHU - minHU)
    npzarray[npzarray>1] = 1
    npzarray[npzarray<0] = 0
    npzarray *= 255

    return (npzarray.astype(int))


def make_mask(center,diam,z,width,height,spacing,origin):
    '''
    '''
    mask = np.zeros([height,width])  #匹配图像

    #定义结节所在的体素范围
    v_center = (center-origin)/spacing
    v_diam = int(diam/spacing[0]+5)
    v_xmin = np.max([0,int(v_center[0]-v_diam)-5])
    v_xmax = np.min([width-1,int(v_center[0]+v_diam)+5])
    v_ymin = np.max([0,int(v_center[1]-v_diam)-5])
    v_ymax = np.min([height-1,int(v_center[1]+v_diam)+5])

    v_xrange = range(v_xmin,v_xmax+1)
    v_yrange = range(v_ymin,v_ymax+1)

    x_data = [x*spacing[0]+origin[0] for x in range(width)]
    y_data = [x*spacing[1]+origin[1] for x in range(height)]

    #结节周围全都填充1
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            if np.linalg.norm(center-np.array([p_x,p_y,z]))<=diam:
                mask[int((p_y-origin[1])/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1.0
    
    return (mask)




luna_path = "E:/LUNA16/src/subset0/"
output_path = "E:/VSCode/lung/npy"
file_list = glob("E:\LUNA16\src\subset0/" + "*.mhd")

# 获取数据中的每一行
def get_filename(case):
    global file_list
    for f in file_list:
        if case in f:
            return (f)


# 节点的位置
df_node = pd.read_csv('E:\LUNA16\src/annotations.csv')
df_node["file"] = df_node["seriesuid"].apply(get_filename)
# df_node: 文件的路径  example：E:\LUNA16\src\subset0\1.3.6.1.4.1.14519.5.2.1……
df_node = df_node.dropna()

# 循环遍历图像文件
#fcount = 0
for fcount, img_file in enumerate(tqdm(file_list)):
    print("Getting mask for image file %s" % img_file.replace(luna_path, ""))
    mini_df = df_node[df_node["file"] == img_file]  # 得到所有结节
    if(len(mini_df) > 0):    #跳过没有结节的文件
        # biggest_node = np.argsort(mini_df["diameter_mm"].values)[-1]  #返回最大的索引值
        # node_x = mini_df["coordX"].values[biggest_node]
        # node_y = mini_df["coordY"].values[biggest_node]
        # node_z = mini_df["coordZ"].values[biggest_node]
        # diam = mini_df["diameter_mm"].values[biggest_node]

        #读取数据
        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img)   #(z,y,x)
        num_z,height,width = img_array.shape    #heightXwidth constitute the transverse plane  
        origin = np.array(itk_img.GetOrigin())  #世界坐标系下的x,y,z(mm)
        spacing = np.array(itk_img.GetSpacing()) #世界坐标中的体素间隔(mm)

        #遍历所有节点
        for node_idx,cur_row in mini_df.iterrows():
            node_x = cur_row["coordX"]
            node_y = cur_row["coordY"]
            node_z = cur_row["coordZ"]
            diam = cur_row["diameter_mm"]

            #保留三个切片
            imgs = np.ndarray([3,height,width],dtype=np.float32)
            masks = np.ndarray([3,height,width],dtype=np.uint8)
            center = np.array([node_x,node_y,node_z])  #结点中心
            v_center = np.rint((center-origin)/spacing) #体素坐标系的结点中心(x,y,z)
            for i,i_z in enumerate(np.arange(int(v_center[2])-1,int(v_center[2])+2).clip(0,num_z-1)):   #clip防止超出z
                mask = make_mask(center,diam,i_z*spacing[2]+origin[2],width,height,spacing,origin)
                masks[i] = mask
                #imgs[i] = img_array[i_z]
                imgs[i] = normalizePlanes(img_array[i_z])
            
            np.save(os.path.join(output_path,"images_%04d_%04d.npy" % (fcount, node_idx)),imgs)
            np.save(os.path.join(output_path,"masks_%04d_%04d.npy" % (fcount, node_idx)),masks)




        





    