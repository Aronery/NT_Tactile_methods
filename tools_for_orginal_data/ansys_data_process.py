import csv
import chardet
import numpy as np
import matplotlib.pyplot as plt
#将excel文件转为图像
root_dir_all="C:/Users/18142\Desktop/ansys_force/file_point_all.xls"
root_point="C:/Users/18142/Desktop/ansys_force/test_1.xls"
csv_file=open(root_dir_all, mode='r', encoding='GB2312')  #utf-16
csv_file_weizhi=open(root_point, mode='r', encoding='GB2312')
#csv_file=open("C:/Users/18142\Desktop/ansys_force/file_point_all_force.txt", mode='r', encoding='GB2312')
csv_reader = csv.reader(csv_file)
csv_reader_weizhi=csv.reader(csv_file_weizhi)
first_column_data = []
second_column_data = []
weizhi_column_data = []



next(csv_reader_weizhi)
for row in csv_reader_weizhi:
    # 将每行的第一列数据加入到列表中
    data = row[0].split("\t")
    #print(data)
    weizhi_column_data.append(int(data[1]))
    #位置信息

csv_file_weizhi.close()
#print(weizhi_column_data)


next(csv_reader)
for row in csv_reader:
    # 将每行的第一列数据加入到列表中

    data=row[0].split("\t")
    data_first=int(data[0])
    data_second = float(data[1])
    first_column_data.append(data_first)
    second_column_data.append(data_second)

csv_file.close()
# print(first_column_data)
# print(second_column_data)

all_force_dict = dict(zip(first_column_data, second_column_data))
#print(all_force_dict)

new_dict = {key: all_force_dict[key] for key in weizhi_column_data if key in all_force_dict}
values_list = [value for value in new_dict.values()]  ###拥有结点数据的列表
#print(new_dict)
print("values_list:")
print(values_list)

matrix = np.zeros((10, 10))
for i in range(0, 100, 10):
    # 计算当前组应该放置的行索引（从最后一行开始）
    row_index = 9 - i // 10
    # 将当前组的元素放入矩阵的对应行
    matrix[row_index, :] = values_list[i:i+10]

#matrix = np.array(matrix).reshape((10, 10))
print("mat:")
#matrix = [values_list[i:i+10] for i in range(0, 100, 10)]
print(matrix)
#np.savetxt('data_array/data_2.txt', matrix, delimiter='\t',fmt='%.3f')


def point_0_255_line(array):
    """
    :param array: 采用的是线性
    :return: 一个（0,255）范围内的数据
    """
    min_val = np.min(array)
    max_val = np.max(array)
    # 计算缩放系数
    scale = 255.0 / (max_val - min_val)
    # 进行线性缩放
    scaled_data = np.round((array - min_val) * scale)
    # 确保缩放后的数据在0到255的范围内
    scaled_data = np.clip(scaled_data, 0, 255).astype(np.uint8)
    return scaled_data

target=point_0_255_line(matrix)
#print(target)

cmap = 'winter'

fig = plt.figure()
ax_1 = fig.add_subplot()
ax_1.imshow(target, cmap=cmap)
ax_1.set_xticks([])
ax_1.set_yticks([])
ax_1.set_title('X axis')
plt.savefig('data_array/HR_2.png')
# plt.savefig('H:/python_project/NT-SRGAN and NT-SRCNN/utility/test_2.png')
plt.show()


# import chardet
#
# with open(root_point, 'rb') as f:
#     result = chardet.detect(f.read())  # 读取一定量的数据进行编码检测
#
# print(result['encoding'])  # 打印检测到的编码
