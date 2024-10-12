#!/usr/bin/python3
from statistics import mode
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import skimage.metrics as metrics
from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from math import log10
import random

from utility.tactileDataLoader import TactileDataLoader

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(42)


dirname, _ = os.path.split(os.path.abspath(sys.argv[0]))
root_path = os.path.dirname(dirname) + '/'


# 加载训练数据
# dataset_file = root_path + 'dataset/TSR_data/train/'              # 原始数据集
# dataset_file = root_path + 'dataset/TSR_data_0/train/'              # 原始数据集
dataset_file = root_path + 'dataset/TSR_data_x5/test/'              # 新数据集
tactile_dataset = TactileDataLoader(dataset_file)

batch_size = 1
test_batch_size = 1

dataset_len = tactile_dataset.__len__()
# train_size = int(dataset_len * 0.8)
# test_size = int(dataset_len - train_size) 

train_size = 1
test_size = dataset_len - train_size

train_dataset, test_dataset = torch.utils.data.random_split(tactile_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
print(train_loader.__len__())

lr_size = (4, 4)
hr_size = (20,20)

# test
linear_avg_psnr = 0
linear_avg_ssim = 0

cubic_avg_psnr = 0
cubic_avg_ssim = 0

for lr_data, hr_data in test_loader:
    bs = lr_data.shape[0]
    lr_data_z = lr_data[:,2:3,:,:].numpy()
    hr_data_z = hr_data[:,2:3,:,:].numpy()

    linear_psnr = 0
    linear_ssim = 0
    
    cubic_psnr = 0
    cubic_ssim = 0
    for i in range(bs):
        single_lr_data_z = lr_data_z[i,0,:,:]
        single_hr_data_z = hr_data_z[i,0,:,:]
        sing_linear_hr_z = cv2.resize(single_lr_data_z, (20, 20), interpolation=cv2.INTER_LINEAR)
        sing_cubic_hr_z = cv2.resize(single_lr_data_z, (20, 20), interpolation=cv2.INTER_NEAREST)
        
        # psnr += metrics.simple_metrics.peak_signal_noise_ratio(single_model_out, single_hr_data_z, data_range=1)
        # ssim += metrics.structural_similarity(single_model_out, single_hr_data_z, data_range=1)
        data_range = single_hr_data_z.max()
        # data_range = 1
        linear_psnr += metrics.simple_metrics.peak_signal_noise_ratio(sing_linear_hr_z, single_hr_data_z, data_range=data_range)
        linear_ssim += metrics.structural_similarity(sing_linear_hr_z, single_hr_data_z, data_range=data_range)
        cubic_psnr += metrics.simple_metrics.peak_signal_noise_ratio(sing_cubic_hr_z, single_hr_data_z, data_range=data_range)
        cubic_ssim += metrics.structural_similarity(sing_cubic_hr_z, single_hr_data_z, data_range=data_range)
        print("linear PSNR {:.4f} , SSIM {:.4f} | CUBIC PSNR {:.4f}, SSIM {:.4f}"\
            .format(linear_psnr, linear_ssim, cubic_psnr, cubic_ssim))

    linear_psnr /= bs
    linear_ssim /= bs
    cubic_psnr /= bs
    cubic_ssim /= bs
    
    linear_avg_psnr += linear_psnr
    linear_avg_ssim += linear_ssim

    cubic_avg_psnr += cubic_psnr
    cubic_avg_ssim += cubic_ssim

linear_avg_psnr /= len(test_loader)
linear_avg_ssim /= len(test_loader)

cubic_avg_psnr /= len(test_loader)
cubic_avg_ssim /= len(test_loader)

print("---------------------")

print("linear PSNR {:.4f} , SSIM {:.4f} | CUBIC PSNR {:.4f}, SSIM {:.4f}"\
    .format(linear_avg_psnr, linear_avg_ssim, cubic_avg_psnr, cubic_avg_ssim))


