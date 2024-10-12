import csv
import numpy as np
from itertools import islice
root_path=r'C:/Users/18142/Desktop/raw_data/force_clrcle_O.csv'
save_path=r'C:/Users/18142/Desktop/save_data/new_2.npy'
import pandas as pd
def getdata_fromcircle(raw_path):
    with open(raw_path) as file:
        reader = csv.reader(file)
        row_number = 2 # 第三行（索引为2）
        specific_row = list(islice(reader, row_number, row_number + 16))
        #print(specific_row[:,0])
        list_shape = np.array(specific_row)
        #print(list_shape.shape)
        #print(list_shape[:,0])
        arr = np.array([list_shape[:,0],list_shape[:,1], list_shape[:,2]])
        target=arr.reshape(3,4,4)
        #print(arr.shape)
        #print(arr)
        # print(target.shape)
        # print(target)
        # np.save(save_path,target,allow_pickle = True,fix_imports = True)
        return target

if __name__ == "__main__":
    #getdata_fromcircle(root_path,save_path)
    # data = np.load(save_path, allow_pickle=True)
    # print(data[0])
    target=getdata_fromcircle(root_path)
    print(target.shape)