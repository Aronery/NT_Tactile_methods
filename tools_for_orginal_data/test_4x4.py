import random
try:
    from .RegTactileData import RegTactileData
except:
    from RegTactileData import RegTactileData

from PIL import ImageDraw
from PIL import Image
import numpy as np
import math
# 创建一个4x4的白色图像
image = Image.new('RGB', (4, 4), 'white')
draw = ImageDraw.Draw(image)
root="H:/python_project/tools for orginal data/dataset/all_40x40/train/_2_rot_0.npy"

#print(float_array.dtype)



def point_0_255_line(array):
    """
    :param array: 采用的是线性
    :return: 一个（0,255）范围内的数据
    """
    min_val = np.min(array)
    max_val = np.max(array)
    # 计算缩放系数
    scale = 255.0 / (max_val - min_val)
    # 进行线性缩放
    scaled_data = np.round((array - min_val) * scale)
    # 确保缩放后的数据在0到255的范围内
    scaled_data = np.clip(scaled_data, 0, 255).astype(np.uint8)
    return scaled_data
def point_0_255_Z_score(data):
    # 计算均值和标准差
    mean = np.mean(data)
    std = np.std(data)
    # 进行Z-score变换
    z_score = (data - mean) / std
    # 将Z-score映射到(0, 255)范围
    # 使用线性映射
    min_val = np.min(z_score)
    max_val = np.max(z_score)
    scale = 255.0 / (max_val - min_val)
    scaled_data = np.round((z_score - min_val) * scale)
    # 确保值在0到255的范围内
    scaled_data = np.clip(scaled_data, 0, 255).astype(np.uint8)
    return scaled_data

def point_0_255_mi_score(data):
    """
    
    :param data: array
    :return: 0-255
    """
    min_val = np.min(data)
    shifted_data = data - min_val

    # 应用指数变换
    base = 2  # 指数的基数，可以调整
    transformed_data = np.exp(shifted_data * base)

    # 将数据映射到0到255之间的范围
    min_val = np.min(transformed_data)
    max_val = np.max(transformed_data)
    mapped_data = (transformed_data - min_val) / (max_val - min_val) * 255

    # 打印映射后的数据
    return mapped_data



def process_LR(array,method_line_or_score):#

    """
    1：线性变化
    2：高斯变化
    3：幂律变换
    :param array: 3x4x4 array
    :return: 3x4x4 的整理过的数据
    """
    #array = np.load(root, allow_pickle=True)
    # print(array.shape)
    # print(array[0][0][0])
    # print(array)
    array = array.astype(np.float64)  # 转为float类型
    if method_line_or_score==1:
        array_x=point_0_255_line(array[0])
        array_y=point_0_255_line(array[1])
        array_z=point_0_255_line(array[2])
        target = np.zeros(array.shape)
        target[0] = array_x
        target[1] = array_y
        target[2] = array_z

        return target

    elif method_line_or_score==2:
        array_x = point_0_255_Z_score(array[0])
        array_y = point_0_255_Z_score(array[1])
        array_z = point_0_255_Z_score(array[2])
        target = np.zeros(array.shape)
        target[0] = array_x
        target[1] = array_y
        target[2] = array_z

        return target

    elif method_line_or_score==3 :
        array_x = point_0_255_mi_score(array[0])
        array_y = point_0_255_mi_score(array[1])
        array_z = point_0_255_mi_score(array[2])
        target = np.zeros(array.shape)
        target[0] = array_x
        target[1] = array_y
        target[2] = array_z

        return target

    # print("x:",array_x)
    # print("y:",array_y)
    # print("z:",array_z)




if __name__ == '__main__':
    ProcessData = RegTactileData()
    array = np.load(root, allow_pickle=True)
    #print("array:",array[0])
    target_array=process_LR(array,method_line_or_score=3)
    # for i in range(4):
    #     for j in range(4):
    #         value = target_array[0][i][j]
    #         color = (0, 0, int(value))
    #         draw.point((i, j), fill=color)

    ProcessData.plotRegData2D(target_array[0], target_array[1], target_array[2])
    # 保存随机像素图像

    #image.save('4x4_x_line.png')
    #image.save('4x4_x_z_new.png')
