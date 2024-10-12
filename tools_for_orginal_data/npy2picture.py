import os
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
test_file_name = "H:/python_project/Tactile_Pattern_SR/dataset/all_40x40_final/val/"
target_file_train = "H:/python_project/Image-Super-Resolution-via-Iterative-Refinement/dataset/val/lr_4/"
target_file_test = "H:/python_project/Image-Super-Resolution-via-Iterative-Refinement/dataset/val/hr_40/"
sr_file_test="H:/python_project/Image-Super-Resolution-via-Iterative-Refinement/dataset/val/sr_4_40/"
import cv2
def get_npy_filenames(folder_path):
    npy_files = [file for file in os.listdir(folder_path) if file.endswith('.npy')]
    return npy_files

npy_files_data = get_npy_filenames(test_file_name)
print(npy_files_data)
# print(npy_files_data)
for name,i in zip(npy_files_data,range(163,400)):
    data = np.load("H:/python_project/NT-SRGAN and NT-SRCNN/dataset/all_40x40_final/val/"+name, allow_pickle=True).item()
    LR_data,HR_data = data["LR"],data["HR"]
    LR_picture,HR_picture = LR_data[2],HR_data[2]
    img_40x40 = cv2.resize(LR_picture, (40, 40), interpolation=cv2.INTER_CUBIC)
    # 保存LR图像
    lr_filename = f"{i:03d}"+'.png'
    plt.imsave(target_file_train+lr_filename, LR_picture,cmap="winter")
    # 保存HR图像
    hr_filename = f"{i:03d}"+'.png'
    plt.imsave(target_file_test+hr_filename, HR_picture,cmap="winter")

    #
    sr_filename = f"{i:03d}"+'.png'
    plt.imsave(sr_file_test+sr_filename, img_40x40,cmap="winter")
    # fig = plt.figure()
    # ax_1 = fig.add_subplot(121)
    # ax_2 = fig.add_subplot(122)
    #
    # ax_1.imshow(LR_data[2], cmap='winter')
    # ax_2.imshow(HR_data[2], cmap='winter')
    # plt.show()

    # print(HR_data.shape)
    # basename = os.path.splitext(os.path.basename(name))[0]
    # print(basename)
