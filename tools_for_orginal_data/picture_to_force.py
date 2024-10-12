import numpy as np
import PIL as Image
import  cv2


##将高分辨率的图像转化为data矩阵
path="H:\python_project\Image-Super-Resolution-via-Iterative-Refinement\dataset/celebahq_4_40/hr_40/001.png"
def inverse_point_0_255_linear(scaled_data):
    """
    :param scaled_data: 经过线性缩放的数据，范围在(0, 255)内
    :param original_min: 原始数据的最小值
    :param original_max: 原始数据的最大值
    :return: 转换回原始范围的数据
    """
    # 计算原始数据的范围
    original_min=np.min(scaled_data)
    original_max=np.max(scaled_data)

    target_min = -1.2
    target_max = 5.4
    slope = (target_max - target_min) / (original_max - original_min)
    intercept = target_min - slope * original_min
    transformed_data = slope * scaled_data + intercept
    rounded_data = np.round(transformed_data, 3)
    return rounded_data


def picture2force(path) :
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image_array = np.array(image)
    return image_array

array=picture2force(path)
# print(array)

data=inverse_point_0_255_linear(array)
print(data)
np.savetxt('data_array/data_1.txt', data, delimiter='\t',fmt='%.3f')