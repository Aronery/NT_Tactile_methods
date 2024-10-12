import os
try:
    from .RegTactileData import RegTactileData
except:
    from RegTactileData import RegTactileData

# def genData(raw_data_file, ProcessData):
#     obj_name, suffix_dot = os.path.splitext(raw_data_file)
#     data_type = obj_name.split('/')[-1].split('_')[0]
#
#     data_x, data_y, data_z = ProcessData.readData(raw_data_file)
#     #print(data_x)
#     #print(data_x.shape)
#     data_x, data_y, data_z = ProcessData.thresholdFilter(data_x, data_y, data_z, thresholdScale=0.95)
#     # print(data_x)
#     # print(data_x.shape)
# if __name__ == '__main__':
#     ProcessData = RegTactileData()
#     root=r"H:\python_project\NT-SRGAN and NT-SRCNN\dataset\\raw_data\\2_B_10x10_3.npy"
# #     genData(root,ProcessData)
# import cv2
# root="H:/python_project/NT-SR3/dataset/celebahq_4_40/hr_40/001.png"
# # 加载图像
# image = cv2.imread(root)
#
# # 检查图像是否加载成功
# if image is None:
#     print("Error: 图像未加载成功，请检查文件路径和文件格式。")
# else:
#     # 图像加载成功，可以进行后续操作
#     # 例如，调整图像大小
#     resized_image = cv2.resize(image, (40, 40))

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# 图片路径
# lr="H:/python_project/NT-SR3/experiments/sr_ffhq_240516_173805/results/1334/40000_3_lr.png"
# hr="H:/python_project/NT-SR3/experiments/sr_ffhq_240516_173805/results/1334/40000_3_hr.png"
# sr="H:/python_project/NT-SR3/experiments/sr_ffhq_240516_173805/results/1334/40000_3_sr.png"
# inf="H:/python_project/NT-SR3/experiments/sr_ffhq_240516_173805/results/1334/40000_3_inf.png"

lr="H:/python_project/NT-SR3/dataset/celebahq_4_40/lr_4/146.png"
hr="H:/python_project/NT-SR3/dataset/celebahq_4_40/lr_4/147.png"
sr="H:/python_project/NT-SR3/dataset/celebahq_4_40/lr_4/148.png"
inf="H:/python_project/NT-SR3/dataset/celebahq_4_40/lr_4/145.png"
image_paths = [lr, hr, sr,inf]

# 文本标识
labels = ['4x4 LR', 'ground truth', '40x40 S R',"inf"]


# 读取图片
img1 = mpimg.imread(lr)
img2 = mpimg.imread(hr)
img3 = mpimg.imread(sr)
img4 = mpimg.imread(inf)
# 创建一个新的图形
plt.figure()

# 添加第一个子图
plt.subplot(1, 4, 1)  # (rows, columns, panel number)
plt.imshow(img1)
plt.title(labels[0])
plt.xticks([])  # 去掉x轴
plt.yticks([])  # 去掉y轴
plt.axis('off')  # 关闭坐标轴

# 添加第二个子图
plt.subplot(1, 4, 2)
plt.imshow(img2)
plt.title(labels[1])
plt.xticks([])  # 去掉x轴
plt.yticks([])  # 去掉y轴
plt.axis('off')


# 添加第二个子图
plt.subplot(1, 4, 3)
plt.imshow(img3)
plt.title(labels[2])
plt.xticks([])  # 去掉x轴
plt.yticks([])  # 去掉y轴
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(img4)
plt.title(labels[3])
plt.axis('off')
# 显示图形
plt.xticks([])  # 去掉x轴
plt.yticks([])  # 去掉y轴
plt.axis('off')  # 去掉坐标轴

plt.show()


