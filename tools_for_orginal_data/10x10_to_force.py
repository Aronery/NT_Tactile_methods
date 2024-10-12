import csv
import numpy as np
from itertools import islice
import os
import cv2
import test_4x4
from PIL import Image
import matplotlib.pyplot as plt
import dataprocess
import HR_process
try:
    from .RegTactileData import RegTactileData
except:
    from RegTactileData import RegTactileData


ProcessData = RegTactileData()
root_path=r'downsampled_image_40x40_L.png'
save_path=r'C:/Users/18142/Desktop/save_data/new_2.npy'
path="C:/Users/18142/Desktop/save_data/7N_npy/touch_data_8N_0.npy"

def picture2point(root_path):
    """

    :param root_path:
    :return: 一个3x40x40 的矩阵
    """
    #降采样到40x40
    height, width=40,40
    image=HR_process.downsampled_image(root_path)
    new_image = np.zeros((40, 40), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            # 获取当前像素的灰度值
            pixel = image[y, x]
            # 将像素值交换
            new_image[x, y] = pixel

    #image = cv2.imread(root_path)
    #grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    np_array = np.array(new_image)
    array_3x40x40 = np.full((3, 40, 40), 7)
    array_3x40x40[0]=np_array
    array_3x40x40[1]=np_array
    array_3x40x40[2]=np_array
    return array_3x40x40

def insert_2_array(LR_root,HR_data):
    # 加载.np文件
    LR_data=test_4x4.process_LR(LR_root, method_line_or_score=1)

    #LR_data = np.load(LR_root,allow_pickle=True)
    #HR_data = np.load(HR_root)

    dataset = {'LR': LR_data, 'HR': HR_data}

    # 打印加载的数组以确认

    np.save("H:/python_project/circle_picture/circle_picture.npy", arr=dataset, allow_pickle=True)

    # 关闭打开的文件

def process_alldata(folder_path,HR_data):
    """
    处理一个文件夹里的所有npy文件
    :return:
    """
    file_list = os.listdir(folder_path)
    for file_name,i in zip(file_list,range(72)):
        file_path = os.path.join(folder_path, file_name)
        LR_data=test_4x4.process_LR(file_path,method_line_or_score=1)
        dataset = {'LR': LR_data, 'HR': HR_data}
        np.save("H:/python_project/tools for orginal data/dataset/C_40x40/train/"+"9N_"+str(i)+"_"+".npy", arr=dataset, allow_pickle=True)



def show_picture(HR_path,Dd_path):

    cmap = 'winter'
    data_x=picture_(HR_path)
    fig = plt.figure()
    ax_1 = fig.add_subplot(121)
    ax_2 = fig.add_subplot(122)

    # ax_1.imshow(data_x, vmin=xy_vmin, vmax=xy_vmax, cmap=cmap)
    # ax_2.imshow(data_y, vmin=xy_vmin, vmax=xy_vmax, cmap=cmap)
    # ax_3.imshow(data_z, vmin=z_vmin, vmax=z_vmax, cmap=cmap)
    ax_1.imshow(data_x[0], cmap=cmap)
    ax_2.imshow(Dd_path, cmap=cmap)


    ax_1.set_xticks([])
    ax_1.set_yticks([])
    ax_2.set_xticks([])
    ax_2.set_yticks([])


    ax_1.set_title('raw picture ')
    ax_2.set_title('40x40 picture')
    plt.savefig('./HR.png')
    # plt.savefig('H:/python_project/NT-SRGAN and NT-SRCNN/utility/test_2.png')
    plt.show()

if __name__ == "__main__":
    C_40x40_path="C:/Users/18142/Desktop/raw_data/9N/9N_0.jpeg"
    HR_target=picture2point(C_40x40_path)
    #insert_2_array(path,HR_target)
    path_folder="C:/Users/18142/Desktop/save_data/9N_npy"
    process_alldata(path_folder,HR_target)

##########test#################333
    # test_file_name = 'dataset/all_40x40/train/_28_rot_1.npy'
    # root="C:/Users/18142/Desktop/raw_data/picture/28.jpeg"
    # data = np.load(test_file_name, allow_pickle=True).item()
    # lr_data, hr_data = data['LR'], data['HR']
    # print(hr_data.shape)
    # print(lr_data.shape)
    # print(lr_data)
    # #ProcessData.plotRegData2D(hr_data[0], hr_data[1], hr_data[2])
    # #ProcessData.plotRegData2D(lr_data[0], lr_data[1], lr_data[2])
    # #ProcessData.plotRegData3D(hr_data[0], hr_data[1], hr_data[2])
    # #ProcessData.plotRegData3D(lr_data[0], lr_data[1], lr_data[2])
    # show_picture(root,hr_data[0])


