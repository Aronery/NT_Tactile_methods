#!/usr/bin/python3
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import cv2
import skimage.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 128
criterion = nn.MSELoss().cuda()
    
dirname, _ = os.path.split(os.path.abspath(sys.argv[0]))
root_path = os.path.dirname(dirname) + '/'
#root_path="H:/python_project/NT-SRGAN and NT-SRCNN/"
#dataset_file = root_path + 'dataset/TSR_data_x5/test/'              # 原始数据集
#dataset_file = root_path + 'dataset/C_40x40/train/'              # 原始数据集
dataset_file = root_path + 'dataset/all_40x40_final/train/'
#dataset_file = root_path + 'dataset/all_40x40_e/test/'
tactile_dataset = TactileDataLoader(dataset_file)
train_loader = DataLoader(tactile_dataset, batch_size=batch_size, shuffle=False)

print(train_loader.__len__())

lr_size = (4, 4)
hr_size = (40,40)


#path = root_path + 'pth/srcnn_pth/srcnn_dataset_2_lay_num_6/srcnn_epoch_299.pth'
#path = root_path + 'pth/srgan_pth/srgan_data_0.3/srgan_epoch_final_240.pth' ##300
path = root_path + 'pth/srgan_pth/srgan_data_0.5/srgan_epoch_C_140.pth'  ##130
#path = root_path + 'pth/srgan_pth/srgan_data_0.5/srgan_epoch_final2_290.pth'
#path = root_path + 'pth/srgan_pth/srgan_data_0.5_new/srgan_epoch_final_210.pth'
#path= root_path + "pth/srcnn_pth/srcnn_data_0.25/srcnn_epoch_all_250.pth"
#path= root_path + "pth/srcnn_pth/srcnn_data_0.25/srcnn_epoch_250.pth"
#path= root_path + "pth/srcnn_pth/srcnn_data_0.25/srcnn_epoch_final_450.pth"
model = torch.load(path)


fig = plt.figure()
ax1 = fig.add_subplot(141)
ax2 = fig.add_subplot(142)
ax3 = fig.add_subplot(143)
ax4 = fig.add_subplot(144)
# ax5 = fig.add_subplot(155)

for lr_data, hr_data in train_loader:
    #44比较好，test的7比较好还有11
    data_index = 39#data_index = 10   13   18
    single_lr_data_x = lr_data[data_index, 0, :].numpy()
    single_lr_data_y = lr_data[data_index, 1, :].numpy()
    single_lr_data_z = lr_data[data_index, 2, :].numpy()
    single_hr_data_z = hr_data[data_index,2:3,:,:].numpy()

    lr_img_zyx = np.array([single_lr_data_x, single_lr_data_y, single_lr_data_z])
    scaled_image = cv2.resize(single_lr_data_z, (40, 40), interpolation=cv2.INTER_LINEAR)
    input_data = torch.from_numpy(np.array(lr_img_zyx))
    input_data = input_data.unsqueeze(0) # 加一个维度
    input_data = input_data.type(torch.float32).cuda()
    ###
    input_data=input_data.to(device) #把tensor数据放到同一个gpu中
    model_out = model(input_data)
    model_out = model_out.cpu().detach().numpy()[0,0,:,:]
    cmap="winter"
    #
    # ax1.imshow(single_lr_data_z, cmap=cmap)            # 原始4x4的图片
    # ax2.imshow(single_lr_data_y, cmap=cmap)
    # ax3.imshow(single_lr_data_z,  cmap=cmap)  # 模型输出图片
    # ax4.imshow(single_hr_data_z[0, :, :], cmap=cmap)  # 真值


    ax1.imshow(single_lr_data_z, cmap=cmap)
    ax2.imshow(scaled_image, cmap=cmap)                               # 插值图片
    #ax3.imshow(model_out, vmin=0, vmax=2, cmap='Greys')    # 模型输出图片
    ax3.imshow(model_out,  cmap=cmap)  # 模型输出图片
    #ax4.imshow(single_hr_data_z[0,:,:], vmin=0, vmax=2, cmap='Greys')            # 真值
    ax4.imshow(single_hr_data_z[0, :, :], cmap=cmap)  # 真值

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4.set_xticks([])
    ax4.set_yticks([])
    #
    # ax1.set_title('input x')
    # ax2.set_title('input y')
    # ax3.set_title('input z')
    # ax4.set_title('ground truth')


    ax1.set_title('input z')
    ax2.set_title('linear Z')
    ax3.set_title('model out Z')   #'ground truth'  'model out Z'
    ax4.set_title('ground truth')
    plt.savefig('./new_1.png')
    plt.show()

    print("yibaocun")
    

    break
