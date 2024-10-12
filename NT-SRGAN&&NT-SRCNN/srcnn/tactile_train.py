#!/usr/bin/python3
from statistics import mode
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import skimage.metrics as metrics
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from math import log10
import random
from model.net import *

from utility.tactileDataLoader import TactileDataLoader

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(42)

## --- 路径 + debug --- ##
# lay_num = int(sys.argv[1])
# cuda_device_num = int(sys.argv[2])
lay_num = 6
cuda_device_num = 1
# cuda_device_num = -1
per_data = 0.25
# TRAIN_NAME = "srcnn_dataset_2_lay_num_" + str(lay_num) + '_epoch_1000_1e-3_new_data'
TRAIN_NAME = "srcnn_data_0.25"

dirname, _ = os.path.split(os.path.abspath(sys.argv[0]))
root_path = os.path.dirname(dirname) + '/'
root_log = root_path + 'logs/srcnn/' + TRAIN_NAME + '/'
root_pth = root_path + 'pth/srcnn_pth/' + TRAIN_NAME + '/'

if not os.path.exists(root_pth):
    os.makedirs(root_pth)

writer = SummaryWriter(root_log)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(cuda_device_num)
print(TRAIN_NAME, " | CUDA device : ", cuda_device_num)


################ 模型超参数 #####################
batch_size = 32
test_batch_size = 8
epochs = 300
weight_decay = 1e-4
# model = TactileSRCNN_MSRN_in().cuda()       
model = SRCNN_MSRN(feature_layers_num=lay_num, is_init=True).cuda()
#model = SRCNN_MSRN(feature_layers_num=lay_num, is_init=True)
is_nor = False  
scale_value = 1000
scale_value_z = 1000
scale_value_xy = 1000
##################################################

criterion = nn.MSELoss().cuda()
    
# 加载训练数据
# dataset_file = root_path + 'dataset/TSR_data/train/'              # 原始数据集
dataset_file_train = root_path + 'dataset/all_40x40_final/train/'              # 新数据集
dataset_file_test = root_path + 'dataset/all_40x40_final/test/'
tactile_dataset_train = TactileDataLoader(dataset_file_train)
tactile_dataset_test =  TactileDataLoader(dataset_file_test)

dataset_len_train = tactile_dataset_train.__len__()
dataset_len_test = tactile_dataset_test.__len__()
train_size = int(dataset_len_train )
test_size = int(dataset_len_test )
train_dataset, test_dataset = tactile_dataset_train, tactile_dataset_test
# train_dataset, test_dataset = torch.utils.data.random_split(tactile_dataset, [train_size, test_size])

train_size_A = int(per_data*train_size)
train_size_B = int(train_size-train_size_A)
train_dataset_A, test_dataset_B = torch.utils.data.random_split(train_dataset, [train_size_A, train_size_B])


train_loader = DataLoader(train_dataset_A, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

# tactile_dataset = TactileDataLoader(dataset_file)
#
# dataset_len = tactile_dataset.__len__()
# train_size = int(dataset_len * 0.8)
# test_size = int(dataset_len - train_size)
# train_dataset, test_dataset = torch.utils.data.random_split(tactile_dataset, [train_size, test_size])
#
# train_size_A = int(per_data*train_size)
# train_size_B = int(train_size-train_size_A)
# train_dataset_A, test_dataset_B = torch.utils.data.random_split(train_dataset, [train_size_A, train_size_B])
#
#
# train_loader = DataLoader(train_dataset_A, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
print(train_loader.__len__())

lr_size = (4, 4)
hr_size = (40,40)

lr = 0.0001
for epoch in range(epochs):
    # if epoch < 10:
    #     lr = 0.01
    # elif epoch < 30:
    #     lr =  0.001
    # elif epoch < 60:
    #     lr =  0.0001
        
    # Adam > Adagrad > SGD > Adadelta
    # optimizer = optim.Adam(model.parameters(),weight_decay=weight_decay)
    optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
    # optimizer.state

    # train
    epoch_loss = 0
    for lr_data, hr_data in train_loader:
        if is_nor:
            lr_data /= scale_value
            hr_data /= scale_value

        #hr_data_z = hr_data
        hr_data_z = hr_data[:,2:3,:,:]  #raw
        input_data = torch.from_numpy(np.array(lr_data))
        hr_data_z = torch.from_numpy(np.array(hr_data_z))
        ##使用
        input_data = input_data.type(torch.float32).cuda()
        hr_data_z = hr_data_z.type(torch.float32).cuda()

        # input_data = input_data.type(torch.float32)
        # hr_data_z = hr_data_z.type(torch.float32)

        optimizer.zero_grad()
        model_out = model(input_data)
        loss = criterion(model_out, hr_data_z)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    print("===> Epoch[{}]: Loss: {:.4f} | lr = {}".format(epoch, epoch_loss, lr))
    writer.add_scalar("train loss", epoch_loss, epoch)

    # test
    test_loss = 0
    avg_psnr = 0
    avg_ssim = 0
    for lr_data, hr_data in test_loader:
        if is_nor:
            lr_data /= scale_value
            hr_data /= scale_value

        hr_data_z = hr_data[:,2:3,:,:]

        input_data = torch.from_numpy(np.array(lr_data))
        hr_data_z = torch.from_numpy(np.array(hr_data_z))
        
        input_data = input_data.type(torch.float32).cuda()
        hr_data_z = hr_data_z.type(torch.float32).cuda()

        model_out = model(input_data)
        loss = criterion(model_out, hr_data_z)
        test_loss += loss.item()

        model_out = model_out.cpu().detach().numpy()
        hr_data_z = hr_data_z.cpu().detach().numpy()
        psnr = 0
        ssim = 0
        for i in range(model_out.shape[0]):
            single_model_out = model_out[i,0,:,:]
            single_hr_data_z = hr_data_z[i,0,:,:]
            # psnr += metrics.simple_metrics.peak_signal_noise_ratio(single_model_out, single_hr_data_z, data_range=1)
            # ssim += metrics.structural_similarity(single_model_out, single_hr_data_z, data_range=1)
            psnr += metrics.simple_metrics.peak_signal_noise_ratio(single_model_out, single_hr_data_z, data_range=single_hr_data_z.max())
            ssim += metrics.structural_similarity(single_model_out, single_hr_data_z, data_range=single_hr_data_z.max())
        psnr /= model_out.shape[0]
        ssim /= model_out.shape[0]
        
        avg_psnr += psnr
        avg_ssim += ssim

    print("--->epoch [{}] | Test Loss {:.4f} Avg. PSNR: {:.4f} dB | Avg. SSIM: {:.4f} ".format(epoch, test_loss, avg_psnr / len(test_loader), avg_ssim / len(test_loader)))
    writer.add_scalar("test loss", test_loss, epoch)
    writer.add_scalar("test PSNR", avg_psnr / len(test_loader), epoch)
    writer.add_scalar("test SSIM", avg_ssim / len(test_loader), epoch)


    if epoch > 90 and epoch%50==0:
        torch.save(model, root_pth + '/' + 'srcnn_epoch_final_' + str(epoch)+'.pth')
        #torch.save(model, root_pth + '/' + '.pth')
    # torch.save(model, root_pth + '/' + 'srcnn_epoch_' + str(epoch)+'.pth')

