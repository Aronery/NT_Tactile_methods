
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
import network

model =  SRCNN_MSRN(feature_layers_num=6, is_init=True)

# 针对有网络模型，但还没有训练保存 .pth 文件的情况
input = torch.randn(1, 3, 224, 224)  # 随机生成一个输入
modelpath = "./demo.onnx"  # 定义模型结构保存的路径
torch.onnx.export(model, input, modelpath)  # 导出并保存
netron.start(modelpath)

# #  针对已经存在网络模型 .pth 文件的情况
# import netron
#
# modelpath = "./demo.onnx"  # 定义模型数据保存的路径
# netron.start(modelpath)  # 输出网络结构