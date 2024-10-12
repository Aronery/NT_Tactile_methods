#!/usr/bin/python3
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import stat
import copy
import skimage.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from tqdm import tqdm
from math import log10
import random
import time
from model.net import *
from utility.tactileDataLoader import TactileDataLoader

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

def gassFunc(data, mu=0,sd=10):
    coef=1/(np.power(2*np.pi, 0.5)*sd)
    powercoef = (np.power((data-mu), 2)) / (2*np.power(sd,2))
    return coef * np.exp(-1*powercoef)

def gassFilter1D(data, mu=0, sd=2, xFactor=1, yFactor=1):
    """
    data.size = 10x10
    -------> x axis
    |
    |
    y axis
    """
    gassFactor = np.ones(data.shape)
    # xFactor /= gassFunc(0,mu,sd)
    # yFactor /= gassFunc(0,mu,sd)
    xlen, ylen = data.shape[0], data.shape[1]
    # x axis gass
    # for i in range(xlen):
    #     gassFactor[i][0:ylen] *= gassFunc(i,mu,sd)
    # return data - np.multiply(data, dataFactor)
    for i in range(xlen):
        for j in range(ylen):
            gassFactor[i][j] += gassFunc(i,mu,sd) + gassFunc(j,mu,sd)
    # gassFactor /= (xFactor + yFactor) 
    gassFactor = (gassFactor - gassFactor.min()) / (gassFactor.max() - gassFactor.min())
    return np.multiply(data, (xFactor + yFactor)*0.5*gassFactor)+ np.multiply(data, (1-gassFactor))

def TactileMean(downData, upData):
    """
    data must be 1-array
    """
    len = downData.shape[0]
    tmp = 0
    for i in range(len):
        tmp += upData[i] / downData[i]
    return tmp/len

def TactileFilter(hrData):
    """
    hrData.size() = 40x40 MxI,NxJ
    """
    I = 10
    blockSize = (4,4)
    Factor = np.ones((blockSize))
    hFactor = np.ones((blockSize))
    vFactor = np.ones((blockSize))
    for idx in range(blockSize[0]):
        hFactor[idx][1] = TactileMean(hrData[0][I*idx:I*(idx+1),   I],hrData[0][I*idx:I*(idx+1),  I-1])
        hFactor[idx][2] = TactileMean(hrData[0][I*idx:I*(idx+1), 2*I],hrData[0][I*idx:I*(idx+1),2*I-1])
        hFactor[idx][3] = TactileMean(hrData[0][I*idx:I*(idx+1), 3*I],hrData[0][I*idx:I*(idx+1),3*I-1])

        vFactor[1][idx] = TactileMean(hrData[0][  I, I*idx:I*(idx+1)],hrData[0][  I-1, I*idx:I*(idx+1)])
        vFactor[2][idx] = TactileMean(hrData[0][2*I, I*idx:I*(idx+1)],hrData[0][2*I-1, I*idx:I*(idx+1)])
        vFactor[3][idx] = TactileMean(hrData[0][3*I, I*idx:I*(idx+1)],hrData[0][3*I-1, I*idx:I*(idx+1)])
    for i in range(blockSize[0]):
        for j in range(i):
            hFactor[0][i] *= hFactor[0][j]
            hFactor[1][i] *= hFactor[1][j]
            hFactor[2][i] *= hFactor[2][j]
            hFactor[3][i] *= hFactor[3][j]

            vFactor[i][0] *= vFactor[j][0]
            vFactor[i][1] *= vFactor[j][1]
            vFactor[i][2] *= vFactor[j][2]
            vFactor[i][3] *= vFactor[j][3]

    # hFactor[hFactor==1] = 0
    # vFactor[hFactor==1] = 0
    # Factor = (hFactor + vFactor)*0.5
    # Factor[0][0] = 1

    for i in range(blockSize[0]):
        for j in range(blockSize[1]):
            # hrData[0][i*I:(i+1)*I, j*I:(j+1)*I] = gassFilter1D(hrData[0][i*I:(i+1)*I, j*I:(j+1)*I],
            # xFactor=hFactor[i][j], yFactor=vFactor[i][j])
            hrData[0][i*I:(i+1)*I, j*I:(j+1)*I] *= vFactor[i][j]
            hrData[0][i*I:(i+1)*I, j*I:(j+1)*I] *= hFactor[i][j]

    print(hFactor)
    # print(vFactor)
    return hrData



# 模型超参数
res = 10
batch_size = 32
#batch_size = 32
test_batch_size = 8
#test_batch_size = 8
criterion = nn.MSELoss().cuda()
# criterion = nn.L1Loss().cuda()
# epochs 不易取过大，过大的epoch会导致过拟合。
epochs = 60
interpolation_method = cv2.INTER_LINEAR
gaussKernel = 9
gaussSigma = 100



    
# 加载训练数据
#dirname, _ = os.path.split(os.path.abspath(sys.argv[0]))
dirname = r"/home/hedazhong/tmp/pycharm_project_848/"
#dataset_file = '/../dataset/tactiledataset/10x10/train/'       # 原始数据集
#dataset_file = '/../dataset/TSR_data_x5/'
#dataset_file = 'dataset/TSR_data_x5/train/'
dataset_file = 'dataset/C_40x40/train/'
# dataset_file = '/../dataset/tactiledataset/10x10_E/'       # 增强数据集
os.chmod(dirname+dataset_file, stat.S_IRWXU)
tactile_dataset = TactileDataLoader(dirname+dataset_file)

dataset_len = tactile_dataset.__len__()
train_size = int(dataset_len * 0.8)
test_size = int(dataset_len - train_size)
train_dataset, test_dataset = torch.utils.data.random_split(tactile_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

#dirname = r"H:/python_project/NT-SRGAN and NT-SRCNN/"
# dirname, _ = os.path.split(os.path.abspath(sys.argv[0]))
# dataset_file_train = '/dataset/TSR_data_x5/train/'
# dataset_file_test = '/dataset/TSR_data_x5/test/'
# tactile_dataset_train = TactileDataLoader(dirname+dataset_file_train)
# tactile_dataset_test = TactileDataLoader(dirname+dataset_file_test)
#
# train_loader = DataLoader(tactile_dataset_train, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(tactile_dataset_test, batch_size=test_batch_size, shuffle=False)
#
# print(train_loader.__len__())

lr_size = (4, 4)
hr_size = (40,40)


scale_value_z = 1000
scale_value_xy = 1000

fig = plt.figure()
ax1 = fig.add_subplot(151)
ax2 = fig.add_subplot(152)
ax3 = fig.add_subplot(153)
ax4 = fig.add_subplot(154)
# ax5 = fig.add_subplot(155)

for lr_data, hr_data in train_loader:

    # 4 'V'
    data_index = 4

    single_lr_data_x = lr_data[data_index, 0, :].numpy()
    single_lr_data_y = lr_data[data_index, 1, :].numpy()
    single_lr_data_z = lr_data[data_index, 2, :].numpy()
    single_hr_data_x = hr_data[data_index,0,:,:].numpy()
    single_hr_data_y = hr_data[data_index,1,:,:].numpy()
    single_hr_data_z = hr_data[data_index,2,:,:].numpy()
    # single_hr_data_z_o = copy.deepcopy(single_hr_data_z)
    # single_hr_data_z = TactileFilter(single_hr_data_z)

    single_hr_data_x = cv2.GaussianBlur(single_hr_data_x, (gaussKernel,gaussKernel), gaussSigma)
    single_hr_data_y = cv2.GaussianBlur(single_hr_data_y, (gaussKernel,gaussKernel), gaussSigma)
    single_hr_data_z = cv2.GaussianBlur(single_hr_data_z, (gaussKernel,gaussKernel), gaussSigma)


    single_lr_data_x = single_lr_data_x/scale_value_xy
    single_lr_data_y = single_lr_data_y/scale_value_xy
    single_lr_data_z = single_lr_data_z/scale_value_z

    single_hr_data_x = single_hr_data_x/scale_value_z
    single_hr_data_y = single_hr_data_y/scale_value_z
    single_hr_data_z = single_hr_data_z/scale_value_z

    # 3x4x4 原始数据
    lr_img_zyx = np.array([single_lr_data_z, single_lr_data_y, single_lr_data_x])


    in_lr_img_z = cv2.resize(single_lr_data_z, hr_size, interpolation=interpolation_method)
    in_lr_img_y = cv2.resize(single_lr_data_y, hr_size, interpolation=interpolation_method)
    in_lr_img_x = cv2.resize(single_lr_data_x, hr_size, interpolation=interpolation_method)

    in_lr_img_zyx = np.array([in_lr_img_z, in_lr_img_y, in_lr_img_x])
    
    # print(in_lr_img_z.shape)
    # print(in_lr_img_z.shape)

    vmin, vmax = 0, 1
    ax1.imshow(single_lr_data_z,  cmap='Greys')            # 原始4x4的图片
    ax2.imshow(in_lr_img_z,  cmap='Greys')                               # 插值图片
    ax3.imshow(single_hr_data_z, cmap='Greys')    # 模型输出图片
    ax4.imshow(single_hr_data_y,  cmap='Greys')            # 真值

    # ax1.imshow(single_lr_data_z, vmin=vmin, vmax=vmax, cmap='Greys')            # 原始4x4的图片
    # ax2.imshow(in_lr_img_z, vmin=vmin, vmax=vmax, cmap='Greys')                               # 插值图片
    # ax3.imshow(single_hr_data_z, vmin=vmin, vmax=vmax, cmap='Greys')    # 模型输出图片
    # ax4.imshow(single_hr_data_y,  cmap='Greys')            # 真值

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4.set_xticks([])
    ax4.set_yticks([])

    ax1.set_title('input z')
    ax2.set_title('linear interpolation Z')
    ax3.set_title('model out Z')
    ax4.set_title('ground truth')


    plt.savefig('out_C_1.png')
    plt.show()

    break


