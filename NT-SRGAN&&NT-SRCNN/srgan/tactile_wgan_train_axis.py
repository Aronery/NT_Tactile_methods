#!/usr/bin/python3
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

import skimage.metrics as metrics

import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from tqdm import tqdm
from math import log10
import math
import random
import time
from model.gan_net import *
from model.net import *
from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

from utility.tactileDataLoader import TactileDataLoader

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(42)

## --- 路径 + debug --- ##
# TRAIN_NAME = "srgan_dataset_old"
TRAIN_NAME = "srgan_dataset2_new_lay_num_6_yz"

dirname, _ = os.path.split(os.path.abspath(sys.argv[0]))
root_path = os.path.dirname(dirname) + '/'
root_log = root_path + 'logs/srgan/' + TRAIN_NAME + '/'
root_pth = root_path + 'pth/srgan_pth/' + TRAIN_NAME + '/'

if not os.path.exists(root_pth):
    os.makedirs(root_pth)

writer = SummaryWriter(root_log)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
print(device)


################ 模型超参数 #####################
batch_size = 32
test_batch_size = 8
epochs = 300
weight_decay = 1e-4
GNet = SRCNN_MSRN(feature_layers_num=6, input_channel=2, is_init=False).cuda()       
DNet = Dis_Net(LRB_layer_num=3, is_init=False).cuda()
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

dataset_len = tactile_dataset.__len__()
train_size = int(dataset_len * 0.8)
test_size = int(dataset_len - train_size) 
train_dataset, test_dataset = torch.utils.data.random_split(tactile_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

print(train_loader.__len__())

lr_size = (4, 4)
hr_size = (40,40)

mse = nn.MSELoss().cuda()
bce = nn.BCEWithLogitsLoss().cuda()

optimizerG = optim.Adam(GNet.parameters(), weight_decay=weight_decay)
optimizerD = optim.Adam(DNet.parameters(), weight_decay=weight_decay)

val_data = []
for epoch in range(epochs):
    # GNet.train()
    # DNet.train()
    # """
    epoch_loss = 0
    for lr_data, hr_data in train_loader:
        if is_nor:
            lr_data /= scale_value
            hr_data /= scale_value

        hr_data_z = hr_data[:,2:3,:,:]

        # input_data = torch.from_numpy(np.array(lr_data[:, 2:3, :, :]))  # z
        input_data = torch.from_numpy(np.array(lr_data[:, 1:3, :, :]))  # yz
        real_data = torch.from_numpy(np.array(hr_data_z))

        input_data = input_data.type(torch.float32).cuda()
        real_data = real_data.type(torch.float32).cuda()

        fake_data = GNet(input_data)

        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################
        for i in range(2):
            DNet.zero_grad()

            logits_real = DNet(real_data).mean()
            logits_fake = DNet(GNet(input_data)).mean()

            gradient_penalty = compute_gradient_penalty(DNet, real_data, fake_data)

            d_loss = logits_fake - logits_real + 10*gradient_penalty
            # gradient penalty 梯度惩罚，约束D， D 为 1-Lipschitz 函数，保持函数的平滑
            d_loss.backward(retain_graph=True)
            optimizerD.step()

        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###########################
        GNet.zero_grad()
        fake_data = GNet(input_data)
        data_loss = mse(fake_data, real_data)
        epoch_loss += data_loss.item()
        adversarial_loss = -1*DNet(fake_data).mean()
        g_loss = data_loss + 1e-3*adversarial_loss
        g_loss.backward()
        optimizerG.step()
        
        writer.add_scalar("train loss", epoch_loss, epoch)
        
# 
    # """
    with torch.no_grad():
        GNet.eval()
        test_loss = 0
        avg_psnr = 0
        avg_ssim = 0
        for lr_data, hr_data in test_loader:
            if is_nor:
                lr_data /= scale_value
                hr_data /= scale_value

            hr_data_z = hr_data[:,2:3,:,:]

            # input_data = torch.from_numpy(np.array(lr_data[:, 2:3, :, :]))  # z
            input_data = torch.from_numpy(np.array(lr_data[:, 1:3, :, :]))  # yz
            # input_data = torch.from_numpy(np.array(lr_data))
            hr_data_z = torch.from_numpy(np.array(hr_data_z))
            
            input_data = input_data.type(torch.float32).cuda()
            hr_data_z = hr_data_z.type(torch.float32).cuda()

            model_out = GNet(input_data)
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
    
    if epoch > 50 and epoch%10==0:
    # if epoch > 90 and epoch%50==0:
        torch.save(GNet, root_pth  + '/' + 
                   'srgan_epoch_' + str(epoch)+'.pth')
    # torch.save(GNet, root_pth  + '/' + 'srgan_epoch_' + str(epoch)+'.pth')


