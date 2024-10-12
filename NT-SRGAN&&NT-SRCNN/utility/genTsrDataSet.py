#!/usr/bin/python3

import os, sys
from tqdm import tqdm
import random
import numpy as np
from torch.utils.data import Dataset

try:
    from .RegTactileData import RegTactileData
except:
    from RegTactileData import RegTactileData

"""
将raw_data 转化为 TSR_data
----------------------> [0/x]
\  0 1 2 3 4 5 6 7 8 9
0  - - - - - - - - - - 
1  - - - - - - - - - - 
2  - - * * * * * * - - 
3  - - * * * * * * - - 
4  - - * * * * * * - - 
5  - - * * * * * * - - 
6  - - * * * * * * - - 
7  - - * * * * * * - - 
8  - - - - - - - - - - 
9  - - - - - - - - - - 
[2,8) [2,8) 36
[3,7)       16
[4,6)       4
"""
 #10 10 300 48


def delFile(path):
    """
    清空文件夹
    """
    for root, ds, fs, in os.walk(path):
        for f in fs:
            os.remove(path + '/' + f)
            print("remove file:%s ..." % f)


def genData(raw_data_file, tsr_data_path, ProcessData, file_num, res=10, train_ratio=80, travl_num=(4, 6)):
    assert os.path.exists(tsr_data_path)

    train_num, test_num = 0, 0

    obj_name, suffix_dot = os.path.splitext(raw_data_file)
    data_type = obj_name.split('/')[-1].split('_')[0]

    data_x, data_y, data_z = ProcessData.readData(raw_data_file)
    data_x, data_y, data_z = ProcessData.thresholdFilter(data_x, data_y, data_z, thresholdScale=0.95)
    data_x, data_y, data_z = ProcessData.TactileSeq2Single(data_x), ProcessData.TactileSeq2Single(
        data_y), ProcessData.TactileSeq2Single(data_z)
    data_x, data_y, data_z = ProcessData.saclePattern(data_x, data_y, data_z)
    if res == 10:
        if data_type == '2':
            sr_data_x, sr_data_y, sr_data_z = ProcessData.regData_new(data_x), ProcessData.regData_new(
                data_y), ProcessData.regData_new(data_z)
        else:
            sr_data_x, sr_data_y, sr_data_z = ProcessData.regData(data_x), ProcessData.regData(
                data_y), ProcessData.regData(data_z)
    elif res == 5:
        if data_type == '2':
            sr_data_x, sr_data_y, sr_data_z = ProcessData.regData_new_5(data_x), ProcessData.regData_new_5(
                data_y), ProcessData.regData_new_5(data_z)
        else:
            sr_data_x, sr_data_y, sr_data_z = ProcessData.regData_5(data_x), ProcessData.regData_5(
                data_y), ProcessData.regData_5(data_z)
    elif res == 2:
        if data_type == '2':
            sr_data_x, sr_data_y, sr_data_z = ProcessData.regData_new_2(data_x), ProcessData.regData_new_2(
                data_y), ProcessData.regData_new_2(data_z)
        else:
            sr_data_x, sr_data_y, sr_data_z = ProcessData.regData_2(data_x), ProcessData.regData_2(
                data_y), ProcessData.regData_2(data_z)
    else:
        sys.exit()
    sr_data_x, sr_data_y, sr_data_z = ProcessData.smoothPattern(sr_data_x, sr_data_y, sr_data_z, smoth_method=1)
    # data_x.shape    -> (10,10,16)
    # sr_data_x.shape -> (40,40)

    # ProcessData.plotRegData2D(sr_data_x, sr_data_y, sr_data_z)

    for i in range(travl_num[0], travl_num[1]):
        for j in range(travl_num[0], travl_num[1]):
            LR_data_x = data_x[i, j].reshape(4, 4)
            LR_data_y = data_y[i, j].reshape(4, 4)
            LR_data_z = data_z[i, j].reshape(4, 4)
            LR_data = np.array([LR_data_x, LR_data_y, LR_data_z])

            HR_data_x = sr_data_x
            HR_data_y = sr_data_y
            HR_data_z = sr_data_z
            HR_data = np.array([HR_data_x, HR_data_y, HR_data_z])

            dataset = {'LR': LR_data, 'HR': HR_data}
            save_name = str(file_num) + '_' + str(i) + str(j) + '_' + '00' + '.npy'
            if random.randint(0, 100) < train_ratio:
                np.save(file=tsr_data_path + 'train/' + save_name, arr=dataset, allow_pickle=True)
                train_num += 1
            else:
                np.save(file=tsr_data_path + 'test/' + save_name, arr=dataset, allow_pickle=True)
                test_num += 1

            # 镜像
            LR_data_x_mir = np.fliplr(LR_data_x)
            LR_data_y_mir = np.fliplr(LR_data_y)
            LR_data_z_mir = np.fliplr(LR_data_z)
            LR_data = np.array([LR_data_x_mir, LR_data_y_mir, LR_data_z_mir])

            HR_data_x_mir = np.fliplr(HR_data_x)
            HR_data_y_mir = np.fliplr(HR_data_y)
            HR_data_z_mir = np.fliplr(HR_data_z)
            HR_data = np.array([HR_data_x_mir, HR_data_y_mir, HR_data_z_mir])
            save_name = str(file_num) + '_' + str(i) + str(j) + '_' + '10' + '.npy'
            dataset = {'LR': LR_data, 'HR': HR_data}
            if random.randint(0, 100) < train_ratio:
                np.save(file=tsr_data_path + 'train/' + save_name, arr=dataset, allow_pickle=True)
                train_num += 1
            else:
                np.save(file=tsr_data_path + 'test/' + save_name, arr=dataset, allow_pickle=True)
                test_num += 1

            # 旋转
            for rot_index in range(1, 4):
                # 原始图片旋转
                LR_data_rot = np.array([np.rot90(LR_data_x, rot_index),
                                        np.rot90(LR_data_y, rot_index),
                                        np.rot90(LR_data_z, rot_index)])
                HR_data_rot = np.array([np.rot90(HR_data_x, rot_index),
                                        np.rot90(HR_data_y, rot_index),
                                        np.rot90(HR_data_z, rot_index)])
                save_name = str(file_num) + '_' + str(i) + str(j) + '_' + '0' + str(rot_index) + '.npy'
                dataset = {'LR': LR_data_rot, 'HR': HR_data_rot}
                if random.randint(0, 100) < train_ratio:
                    np.save(file=tsr_data_path + 'train/' + save_name, arr=dataset, allow_pickle=True)
                    train_num += 1
                else:
                    np.save(file=tsr_data_path + 'test/' + save_name, arr=dataset, allow_pickle=True)
                    test_num += 1

                # 镜像图片旋转
                LR_data_mir = np.array([np.rot90(LR_data_x_mir, rot_index),
                                        np.rot90(LR_data_y_mir, rot_index),
                                        np.rot90(LR_data_z_mir, rot_index)])
                HR_data_mir = np.array([np.rot90(HR_data_x_mir, rot_index),
                                        np.rot90(HR_data_y_mir, rot_index),
                                        np.rot90(HR_data_z_mir, rot_index)])
                save_name = str(file_num) + '_' + str(i) + str(j) + '_' + '1' + str(rot_index) + '.npy'
                dataset = {'LR': LR_data_mir, 'HR': HR_data_mir}
                if random.randint(0, 100) < train_ratio:
                    np.save(file=tsr_data_path + 'train/' + save_name, arr=dataset, allow_pickle=True)
                    train_num += 1
                else:
                    np.save(file=tsr_data_path + 'test/' + save_name, arr=dataset, allow_pickle=True)
                    test_num += 1

    print(raw_data_file, ", save_success.. train_num = {}, test_num = {} ".format(train_num, test_num))


if __name__ == "__main__":
    dirname, _ = os.path.split(os.path.abspath(sys.argv[0]))
    root_path = os.path.dirname(dirname) + '/'
    #root_path = "H:/python_project/NT-SRGAN and NT-SRCNN/"
    raw_data_path = root_path + 'dataset/raw_data/'
    tsr_data_path = root_path + 'dataset/TSR_data_x5/'
    print(tsr_data_path)
    ProcessData = RegTactileData()

    # #sys.exit()
    # #---- 生成数据 ---- ##
    # res = 10
    # file_num = 3
    # for root, ds, fs, in os.walk(raw_data_path):
    #     for f in fs:
    #         obj_name, suffix_dot = os.path.splitext(f)
    #         if suffix_dot == '.npy':
    #             data_type = obj_name.split('/')[-1].split('_')[0]
    #
    #             ## --- 所有数据 --- ##
    #             genData(raw_data_path + f, tsr_data_path, ProcessData, file_num=file_num)
    #
    #             ## --- 旧的数据 --- ##
    #             # if not data_type == '2':
    #             #   genData(raw_data_path+f, tsr_data_path, ProcessData, file_num=file_num, res=res)
    #             #   file_num += 1
    #
    #             ## --- 旧数据 + 部分新数据 --- ##
    #             # if data_type == '2':
    #             #     ## --- 只包括字母 --- ##
    #             #     data_name = obj_name.split('/')[-1].split('_')[1]
    #             #     if len(data_name) == 1:
    #             #         genData(raw_data_path+f, tsr_data_path, ProcessData, file_num=file_num, res=res)
    #             #         file_num += 1
    #             #     else:
    #             #         pass
    #             # else:
    #             #     genData(raw_data_path+f, tsr_data_path, ProcessData, file_num=file_num, res=res)
    #             #     file_num += 1
    # print("total contact surface = ", file_num)

    ## ---- 清空文件夹 ---- ##
    # delFile(tsr_data_path+'train/')
    # delFile(tsr_data_path+'test/')

    ##---- 测试案例 ---- ##
    #test_file_name = 'train/9_44_01.npy'
    #test_file_name = 'train/1_44_01.npy'
    test_file_name = 'train/1_44_01.npy'

    test_path="H:/python_project/Tactile_Pattern_SR\dataset/all_40x40_final/train/_22_rot_0.npy"
    #test_path="H:/python_project/NT-SRGAN and NT-SRCNN/dataset/all_40x40_e/train/_16_rot_2.npy"
    #data = np.load(tsr_data_path+test_file_name, allow_pickle=True).item()
    data = np.load(test_path, allow_pickle=True).item()
    lr_data, hr_data = data['LR'], data['HR']
    print(hr_data.shape)
    print(lr_data.shape)
    #np.savetxt(r"..\test_lr.txt",lr_data, delimiter=',')
    # ProcessData.plotRegData2D(lr_data[0],lr_data[1],lr_data[2])
    #ProcessData.plotRegData2D(hr_data[0],hr_data[1],hr_data[2],name="HR")
    #ProcessData.plotRegData2D(lr_data[0],lr_data[1],lr_data[2])
    #ProcessData.plotRegData2D(hr_data[0], hr_data[1], hr_data[2])
    ProcessData.plotRegData2D(lr_data[0], lr_data[1], lr_data[2])

    #ProcessData.plotRegData2D(hr_data[0], hr_data[1], hr_data[2] )
    #print(lr_data[0])