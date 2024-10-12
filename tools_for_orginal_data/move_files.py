
import os
import random
import shutil


def move_files(source_folder, destination_folder, file_extension='.npy', move_ratio=0.2):
    # 获取源文件夹中所有符合条件的文件
    files = [f for f in os.listdir(source_folder) if f.endswith(file_extension)]

    # 计算需要移动的文件数量
    num_files_to_move = int(move_ratio * len(files))

    # 从文件列表中随机选择要移动的文件
    files_to_move = random.sample(files, num_files_to_move)

    # 移动文件到目标文件夹
    for file in files_to_move:
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(destination_folder, file)
        shutil.move(source_path, destination_path)
        print(f"Moved {file} to {destination_folder}")


# 源文件夹和目标文件夹的路径
source_folder = 'H:/python_project/NT-SRGAN and NT-SRCNN/dataset/all_40x40_final/train'
destination_folder = 'H:/python_project/NT-SRGAN and NT-SRCNN/dataset/all_40x40_final/val'

# 指定移动比例
move_ratio = 0.1

# 调用函数移动文件
move_files(source_folder, destination_folder, move_ratio=move_ratio)