#!/usr/bin/python3
import fractions
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
lay_num = 6

TRAIN_NAME = 'k_fold_srcnn'

dirname, _ = os.path.split(os.path.abspath(sys.argv[0]))
root_path = os.path.dirname(dirname) + '/'

root_pth = root_path + 'pth/k-fold/' + TRAIN_NAME + '/'
logs_file_name = 'logs/k-fold/'


if not os.path.exists(root_path+logs_file_name):
    os.makedirs(root_path+logs_file_name)
    
if not os.path.exists(root_pth):
    os.makedirs(root_pth)


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cuda_device_num = int(sys.argv[2])
cuda_device_num = 0
torch.cuda.set_device(cuda_device_num)
print(TRAIN_NAME, " | CUDA device : ", cuda_device_num)


################ 模型超参数 #####################
batch_size = 32
test_batch_size = 8
epochs = 300
weight_decay = 1e-4
# model = TactileSRCNN_MSRN_in().cuda()       
model = SRCNN_MSRN(feature_layers_num=lay_num, is_init=True).cuda()       
is_nor = False  
scale_value = 1000
scale_value_z = 1000
scale_value_xy = 1000
##################################################

criterion = nn.MSELoss().cuda()
    
# 加载训练数据
# dataset_file = root_path + 'dataset/TSR_data/train/'              # 原始数据集
# dataset_file = root_path + 'dataset/TSR_data_0/train/'              # 原始数据集
dataset_file = root_path + 'dataset/TSR_data_2/train/'              # 新数据集
tactile_dataset = TactileDataLoader(dataset_file)
all_data_size = tactile_dataset.__len__()
print(all_data_size)
all_train_data = DataLoader(tactile_dataset, batch_size=batch_size, shuffle=True)
k_fold=5
seg_num = int(all_data_size/5) 

lr_size = (4, 4)
hr_size = (40,40)

lr = 0
# for i in range(k_fold):
#     # if i > 3:
#     #     break
    
#     if i  1:
#         continue
 
i = 4
trll = 0
trlr = i * seg_num
vall = trlr
valr = i * seg_num + seg_num
trrl = valr
trrr = all_data_size
print('<---------------------%d th FOLD-------------------------------->' % i)
print("train indices: [%d,%d),[%d,%d), test indices: [%d,%d)" % (trll,trlr,trrl,trrr,vall,valr))

write_comment = '/SRCNN_' + str(i) + 'th_fold'
writer = SummaryWriter(root_path+logs_file_name+write_comment)
k_fold_pth = str(i)+'th_fold'
if not os.path.exists(root_pth+k_fold_pth):
    os.makedirs(root_pth+k_fold_pth)

train_left_indices = list(range(trll,trlr))
train_right_indices = list(range(trrl,trrr))
train_indices = train_left_indices + train_right_indices
val_indices = list(range(vall,valr))

train_dataset = torch.utils.data.dataset.Subset(tactile_dataset,train_indices)
test_dataset = torch.utils.data.dataset.Subset(tactile_dataset,val_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

for epoch in range(epochs):
    optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)

    # train
    epoch_loss = 0
    for lr_data, hr_data in train_loader:
        if is_nor:
            lr_data /= scale_value
            hr_data /= scale_value

        hr_data_z = hr_data[:,2:3,:,:]

        input_data = torch.from_numpy(np.array(lr_data))
        hr_data_z = torch.from_numpy(np.array(hr_data_z))

        input_data = input_data.type(torch.float32).cuda()
        hr_data_z = hr_data_z.type(torch.float32).cuda()

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


    if epoch > 200 and epoch%10==0:
        torch.save(model, root_pth+k_fold_pth + '/' + 'srcnn_epoch_' + str(epoch)+'.pth')    
    # torch.save(model, root_pth + '/' + 'srcnn_epoch_' + str(epoch)+'.pth')

