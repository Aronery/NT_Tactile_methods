import cv2
import numpy as np
import HR_process
import dataprocess
import random
try:
    from .RegTactileData import RegTactileData
except:
    from RegTactileData import RegTactileData
def thresholdFilter(seq_data_x, seq_data_y, seq_data, thresholdScale=0.1):
    """
    阈值滤波
    """
    for i in range(seq_data.shape[0]):
        for j in range(seq_data.shape[1]):
            threshold_data = 0
            for seq_index in range(seq_data.shape[2]):
                if threshold_data < seq_data[i][j][seq_index].mean():
                    threshold_data = seq_data[i][j][seq_index].mean()
            for seq_index in range(seq_data.shape[2]):
                if seq_data[i][j][seq_index].mean() < threshold_data * thresholdScale:
                    seq_data[i][j][seq_index] = np.zeros((seq_data.shape[-1]))
    for i in range(seq_data.shape[0]):
        for j in range(seq_data.shape[1]):
            for seq_index in range(seq_data.shape[2]):
                if seq_data[i][j][seq_index].sum() == 0:
                    seq_data_x[i][j][seq_index] = np.zeros((seq_data.shape[-1]))
                    seq_data_y[i][j][seq_index] = np.zeros((seq_data.shape[-1]))
    return seq_data_x, seq_data_y, seq_data






def smoothPattern(data_x, data_y, data_z, smoth_method=1):  ##修改了滤波器
    # 1 : 高斯滤波 2：双边滤波
    threshold_value=50
    if smoth_method == 1:
        gaussKernel = 3
        gaussSigma = 1
        reg_data_x = cv2.GaussianBlur(data_x, (gaussKernel, gaussKernel), gaussSigma)
        _, binary_channel0 = cv2.threshold(reg_data_x, threshold_value, 255, cv2.THRESH_BINARY)
        reg_data_y = cv2.GaussianBlur(data_y, (gaussKernel, gaussKernel), gaussSigma)
        _, binary_channel1 = cv2.threshold(reg_data_y, threshold_value, 255, cv2.THRESH_BINARY)
        reg_data_z = cv2.GaussianBlur(data_z, (gaussKernel, gaussKernel), gaussSigma)
        _, binary_channel2 = cv2.threshold(reg_data_z, threshold_value, 255, cv2.THRESH_BINARY)
    elif smoth_method == 2:
        d = 0
        sigmaColor = 5
        sigmaSpace = 2
        reg_data_x = cv2.bilateralFilter(data_x.astype(np.float32), d, sigmaColor, sigmaSpace)
        reg_data_y = cv2.bilateralFilter(data_y.astype(np.float32), d, sigmaColor, sigmaSpace)
        reg_data_z = cv2.bilateralFilter(data_z.astype(np.float32), d, sigmaColor, sigmaSpace)
    else:
        reg_data_x = data_x
        reg_data_y = data_y
        reg_data_z = data_z
    return binary_channel0, binary_channel1, binary_channel2

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

if __name__ == '__main__':
    ProcessData = RegTactileData()
    #root ="H:/python_project/NT-SR3/dataset/val/hr_40/001.png"
    #root="/home/hedazhong/zhou_project/diff_SR/dataset/celebahq_4_40/hr_40/001.png"
    root="H:/python_project/NT-SR3/dataset/celebahq_4_40/hr_40/001.png"
    array=picture2point(root)
    x_data,y_data,z_data =array[0],array[1],array[2]
    #print(x_data.shape)
    #x_data,y_data,z_data=thresholdFilter(x_data,y_data,z_data,thresholdScale=200)
    x_data, y_data, z_data=smoothPattern(x_data,y_data,z_data)
    array[0],array[1],array[2]=x_data, y_data, z_data
    ProcessData.plotRegData2D(array[0], array[1], array[2])
    print(array.shape)




