import numpy as np

# #导入npy文件路径位置
# test = np.load('H:\python_project\NT-SRGAN and NT-SRCNN\dataset\\raw_data\\2_B_10x10_2.npy',allow_pickle=True)
#
# print(test)
# print(test.shape)
# from test_4x4 import point_0_255_line
#
# root="H:\python_project\NT-SRGAN and NT-SRCNN\dataset\circle_rawdata\\new.npy"
# array=np.load(root,allow_pickle=True)
# #print(array.shape)
# # print(array[0][0][0])
# # print(array)
# float_array = array.astype(np.float64) #转为float类型
# array_x=point_0_255_line(float_array[0])
# array_y=point_0_255_line(float_array[1])
# array_z=point_0_255_line(float_array[2])
# print(array_x)
# print("_______________________")
# print(array_y)
# print("_______________________")
# print(array_z)

import matplotlib.pyplot as plt

fig,ax=plt.subplots(1,2,dpi=300)

ax[0].text(0.3,0.5,"1st Subplot")
ax[0].set_xticks([])
ax[0].set_yticks([])

ax[1].text(0.3,0.5,"2nd Subplot")
ax[1].set_xticks([])
ax[1].set_yticks([])

fig.suptitle('Figure with 2 subplots',fontsize=16)
plt.show()

