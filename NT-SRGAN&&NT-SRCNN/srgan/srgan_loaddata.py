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
# from model.net import *

from utility.tactileDataLoader import TactileDataLoader

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(42)

log_file_name = 'srganlog_' + \
                str(time.localtime().tm_mon) + '_' + \
                str(time.localtime().tm_mday) + '_' + \
                str(time.localtime().tm_hour) + '_' + \
                str(time.localtime().tm_min) + '.txt'

log_file = open('log/' + log_file_name, 'w')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# 模型超参数
res = 10
batch_size = 24
test_batch_size = 1
criterion = nn.MSELoss().cuda()
epochs = 60
interpolation_method = cv2.INTER_LINEAR
gaussKernel = 5
gaussSigma = 100

log_file.write("=======================================================")
log_file.write(str(device)+'\n')
log_file.write('learning rate = '+str(0.01)+'\n')
log_file.write('train batch size = '+str(batch_size)+'\n')
log_file.write('test batch size = '+str(test_batch_size)+'\n')
log_file.write('loss function  = '+str(criterion)+'\n')
log_file.write('epochs = '+str(epochs)+'\n')

    
# 加载训练数据
dirname, _ = os.path.split(os.path.abspath(sys.argv[0]))
# dataset_file = '/../dataset/tactiledataset/10x10E/train/'
#dataset_file = '/../dataset/tactiledataset/10x10/train/'
dataset_file = '/dataset/TSR_data_x5/train/'
tactile_dataset = TactileDataLoader(dirname+dataset_file)

dataset_len = tactile_dataset.__len__()
train_size = int(dataset_len * 0.8)
test_size = int(dataset_len - train_size) 
train_dataset, test_dataset = torch.utils.data.random_split(tactile_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

print(train_loader.__len__())

lr_size = (4, 4)
hr_size = (40,40)
