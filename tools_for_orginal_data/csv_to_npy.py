import csv
import numpy as np
from itertools import islice
root_path=r'C:/Users/18142/Desktop/raw_data/force_clrcle_C_9N.csv'
save_path=r'C:/Users/18142/Desktop/save_data/9N_npy/'
import pandas as pd

def getdata_fromcircle(raw_path, save_path):
    """

    :param raw_path:
    :param save_path:
    :return:读取csv，批量保存npy数据
    """
    df = pd.read_csv(raw_path,header=None)
    interval_size = 16
    # 计算有多少个区间
    num_intervals = len(df) // (interval_size+2)
    # print(len(df))
    # print(num_intervals)
    # 循环提取每个区间
    for i in range(0,num_intervals):

        start_index = i * (interval_size+2) + 2  # 区间起始索引（从第3行开始）
        end_index = start_index + interval_size
        # end_index = (i + 1) * (interval_size) + 2  # 区间结束索引（到第18行结束）
        # 提取当前区间的数据
        interval_data = df.iloc[start_index:end_index,:].values
        target_inter= np.array([interval_data[:,0],interval_data[:,1], interval_data[:,2]])
        print(target_inter.shape)
        target = target_inter.reshape(3, 4, 4)
        #print(target.shape)
        #print(target)
        np.save(save_path + "touch_data_7N_" + str(i) + ".npy", target)

def getdata_fromcircle(raw_path,save_path):
    with open(raw_path) as file:
        reader = csv.reader(file)
        row_number = 2 # 第三行（索引为2）
        specific_row = list(islice(reader, row_number, row_number + 16))
        #print(specific_row[:,0])
        list_shape = np.array(specific_row)
        print(list_shape.shape)
        print(list_shape[:,0])
        arr = np.array([list_shape[:,0],list_shape[:,1], list_shape[:,2]])
        target=arr.reshape(3,4,4)
        #print(arr.shape)
        #print(arr)
        # print(target.shape)
        # print(target)
        # np.save(save_path,target,allow_pickle = True,fix_imports = True)
        return target




if __name__ == "__main__":
    #getdata_fromcircle(root_path, save_path)
    getdata_fromcircle(root_path,save_path)
    # data = np.load(save_path+"touch_data_" + str(11) + ".npy", allow_pickle=True)
    # print(data)
    # print(data[0])