import re
import numpy as np
import os
import test_4x4
import HR_process
import csv2npy
try:
    from .RegTactileData import RegTactileData
except:
    from RegTactileData import RegTactileData




def extract_number_from_filename(filename):
    number = ''
    for char in filename:
        if char.isdigit():
            number += char
    return number

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


def data_process(picture_file_path,csv_path_path,save_path):
    csv = os.listdir(csv_path_path)
    picture=os.listdir(picture_file_path)
    csv_files = [file for file in csv if file.endswith('.csv')]  ##所有的csv文件
    picture_files = [file for file in picture if file.endswith('.jpeg')]  ##所有的png文件

    csv_files = sorted(csv_files, key=lambda x: int(re.search(r'\d+', x).group()))

    picture_files= sorted(picture_files, key=lambda x: int(re.search(r'\d+', x).group()))

    #print(csv_files)
    #print(picture_files)
    for csv_file,picture_file in zip(csv_files,picture_files):
        number = extract_number_from_filename(picture_file)
        print(csv_file)
        print(picture_file)
        print("*****************************")
        csv_file=csv2npy.getdata_fromcircle(csv_path_path+csv_file) ##变为3X4X4
        #print(csv_file)
        LR_data=test_4x4.process_LR(csv_file,method_line_or_score=1)#转为0-255格式

        HR_data=picture2point(picture_file_path+picture_file)

        for rot_index in range(0,4):
            LR_data_mir = np.array([np.rot90(LR_data[0], rot_index),
                                    np.rot90(LR_data[1], rot_index),
                                    np.rot90(LR_data[2], rot_index)])
            HR_data_mir = np.array([np.rot90(HR_data[0], rot_index),
                                    np.rot90(HR_data[1], rot_index),
                                    np.rot90(HR_data[2], rot_index)])
            dataset = {'LR': LR_data_mir, 'HR': HR_data_mir}
            np.save(save_path + "_" + str(number) + "_rot_"+str(rot_index) + ".npy", arr=dataset, allow_pickle=True)
            print("saving:" + str(number) + "********************************")


if __name__ == "__main__":
    picture_file = "C:/Users/18142/Desktop/raw_data/picture/"
    csv_path = "C:/Users/18142/Desktop/raw_data/file/"
    save_path = "H:/python_project/Tactile_Pattern_SR/dataset/all_40x40_final/train/"
    #print("qwerqwer")
    data_process(picture_file,csv_path,save_path)
    #print("qwerqwer")




    # ########test#################
    # ProcessData = RegTactileData()
    # test_file_name = 'dataset/all_40x40/train/_7_rot_0.npy'
    # #test_file_name="H:/python_project/NT-SRGAN and NT-SRCNN/dataset/all_40x40_e/train/_5_rot_0.npy"
    # data = np.load(test_file_name, allow_pickle=True).item()
    # lr_data, hr_data = data['LR'], data['HR']
    # print(hr_data.shape)
    # print(lr_data.shape)
    # print(lr_data)
    # #ProcessData.plotRegData2D(hr_data[0], hr_data[1], hr_data[2])
    # ProcessData.plotRegData2D(lr_data[0], lr_data[1], lr_data[2])
    # #ProcessData.plotRegData3D(hr_data[0], hr_data[1], hr_data[2])
    # # ProcessData.plotRegData3D(lr_data[0], lr_data[1], lr_data[2])

