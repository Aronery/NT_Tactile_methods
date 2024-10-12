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
torch.cuda.set_device(0)
print(device)

test_batch_size = 1
criterion = nn.MSELoss().cuda()

    
# 加载训练数据
dirname, _ = os.path.split(os.path.abspath(sys.argv[0]))
root_path = os.path.dirname(dirname) + '/'
# dataset_file = root_path + 'dataset/TSR_data_2/test/'              # 新数据集
# dataset_file = root_path + 'dataset/TSR_data_0/test/'              # 新数据集
dataset_file = root_path + 'dataset/TSR_data_x5/test/'              # 新数据集
tactile_dataset = TactileDataLoader(dataset_file)
test_loader = DataLoader(tactile_dataset, batch_size=test_batch_size, shuffle=False)

print(test_loader.__len__())

# sys.exit()


is_nor = True
scale_value = 1000
scale_value_z = 1000
scale_value_xy = 1000

# file_path = 'pth/srcnn_pth/'
# pth_file ='srcnn_dataset_2_lay_num_6/'
# pth_file ='srcnn_dataset_2_lay_num_6_yz/'
# pth_file ='srcnn_dataset_2_lay_num_6_z/'
# pth_file ='srcnn_dataset_x5_lay_num_6/'

file_path = 'pth/srgan_pth/'
# pth_file = 'srgan_dataset_new_lay_num_6/'
# pth_file = 'srgan_dataset2_new_lay_num_6_yz/'
# pth_file = 'srgan_dataset2_new_lay_num_6_z/'
pth_file ='srgan_dataset_x5_lay_num_6/'


file_path = root_path + file_path


print(file_path+pth_file)
test_data = []
for root, ds, fs, in os.walk(file_path+pth_file):
    for f in fs:
        obj_name, suffix_dot = os.path.splitext(f)
        if suffix_dot == '.pth':
            epoch = int(obj_name.split('_')[-1])
            model = torch.load(file_path + pth_file + f).cuda()
            with torch.no_grad():
                model.eval()
                test_loss = 0
                avg_psnr = 0
                avg_ssim = 0
                for lr_data, hr_data in test_loader:
                    hr_data_z = hr_data[:,2:3,:,:]

                    input_data = torch.from_numpy(np.array(lr_data))
                    # input_data = torch.from_numpy(np.array(lr_data[:, 2:3, :, :]))  # z
                    # input_data = torch.from_numpy(np.array(lr_data[:, 1:3, :, :]))  # yz
                    
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

