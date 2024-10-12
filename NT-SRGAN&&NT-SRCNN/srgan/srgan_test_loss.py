#!/usr/bin/python3
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import skimage.metrics as metrics

import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from tqdm import tqdm
from math import log10
import random
import time
from model.gan_net import *

from utility.tactileDataLoader import TactileDataLoader

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 模型超参数
res = 10
test_batch_size = 32
criterion = nn.MSELoss().cuda()

interpolation_method = cv2.INTER_LINEAR
gaussKernel = 9
gaussSigma = 100

    
# 加载训练数据
dirname, _ = os.path.split(os.path.abspath(sys.argv[0]))
dataset_file = '/../dataset/tactiledataset/10x10/test/'       # 原始数据集
tactile_dataset = TactileDataLoader(dirname+dataset_file)
test_loader = DataLoader(tactile_dataset, batch_size=test_batch_size, shuffle=False)

print(test_loader.__len__())

lr_size = (4, 4)
hr_size = (40,40)

is_nor = True
scale_value = 1000
scale_value_z = 1000
scale_value_xy = 1000

file_path = 'pth/srgan_pth/'
log_pth = 'log/srgan/'

pth_file = 'srcnnseqs_logs1_17_14_27/'

test_data = []
for root, ds, fs, in os.walk(file_path+pth_file):
    for f in fs:
        obj_name, suffix_dot = os.path.splitext(f)
        if suffix_dot == '.pth':
            epoch = int(obj_name.split('_')[-1])
            GNet = torch.load(file_path + pth_file + f).cuda()
            with torch.no_grad():
                GNet.eval()
                test_loss = 0
                avg_psnr = 0
                avg_ssim = 0
                for lr_data, hr_data in test_loader:
                    if is_nor:
                        lr_data /= scale_value
                        hr_data /= scale_value

                    hr_data_z = []
                    for data_index in range(lr_data.shape[0]):
                        single_hr_data_z = hr_data[data_index,2:3,:,:].numpy()
                        single_hr_data_z = cv2.GaussianBlur(single_hr_data_z, (gaussKernel,gaussKernel), gaussSigma)
                        hr_data_z.append(np.array(single_hr_data_z))

                    # input_data = torch.from_numpy(np.array(lr_data))
                    # real_data = torch.from_numpy(np.array(hr_data_z))
                    input_data = torch.from_numpy(np.array(lr_data))
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
                        psnr += metrics.simple_metrics.peak_signal_noise_ratio(single_model_out, single_hr_data_z, data_range=1)
                        ssim += metrics.structural_similarity(single_model_out, single_hr_data_z, data_range=1)
                        # print(single_hr_data_z.shape)

                    psnr /= model_out.shape[0]
                    ssim /= model_out.shape[0]
                    
                    avg_psnr += psnr
                    avg_ssim += ssim

                print("--->epoch [{}] | Test Loss {:.4f} Avg. PSNR: {:.4f} dB | Avg. SSIM: {:.4f} ".format(epoch, test_loss, avg_psnr / len(test_loader), avg_ssim / len(test_loader)))
            test_data.append([epoch, avg_psnr / len(test_loader), avg_ssim / len(test_loader)])

test_file_name = log_pth+pth_file+'/' + 'test_data.npy'
np.save(test_file_name, np.array(test_data))

# """