##---- 测试案例 ---- ##
import os, sys
from tqdm import tqdm
import random
import numpy as np
from torch.utils.data import Dataset

try:
    from .RegTactileData import RegTactileData
except:
    from RegTactileData import RegTactileData
import numpy as np
ProcessData = RegTactileData()

dirname, _ = os.path.split(os.path.abspath(sys.argv[0]))
root_path = os.path.dirname(dirname) + '/'
# root_path = "H:/python_project/NT-SRGAN and NT-SRCNN/"
raw_data_path = root_path
save_path=root_path +'dataset/TSR_data_x5/train/1_44_00.npy'
data = np.load(save_path, allow_pickle=True).item()
# np.savetxt(r"..\test_lr.txt",lr_data, delimiter=',')
# ProcessData.plotRegData2D(lr_data[0],lr_data[1],lr_data[2])
# ProcessData.plotRegData2D(hr_data[0],hr_data[1],hr_data[2],name="HR")
# ProcessData.plotRegData2D(lr_data[0],lr_data[1],lr_data[2])
# ProcessData.plotRegData2D(hr_data[0], hr_data[1], hr_data[2],)
# ProcessData.plotRegData2D(lr_data[0], lr_data[1], lr_data[2])
LR_data,HR_data= data['LR'],data['HR']
#print(LR_data)
print(LR_data.shape)
print(HR_data.shape)
#ProcessData.plotRegData2D(data[0], data[1], data[2])