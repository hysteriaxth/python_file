# writen by @LorneM 2019.11.30
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os,sys
sys.path.append('utils')
#   TE_data ——    TE正常工况数据，数据存放在文件d00.txt
#   CSTR_data ——    CSTR 正常工况数据
#   mill_data   ——    磨煤机数据
#   Synthetic_data  ——  仿真数据

prefix = 'dataset/'

class TE_data():
    def __init__(self):
        datapath = prefix + 'TE_dataset/'
        csv_data_train = pd.read_csv(datapath+'d00.csv',header=None)
        self.train_data = np.array(csv_data_train)[:,:52]

        csv_data_test = pd.read_csv(datapath + 'd00_te.csv', header=None)
        self.test_data = np.array(csv_data_test)[:,:52]

class CSTR_data():
    def __init__(self):
        datapath = prefix + 'CSTR/'
        results = np.load(datapath+'results.npy')
        op_param_records = np.load(datapath + 'op_param_records.npy')
        self.data = np.concatenate((results,op_param_records),axis=1)

class mill_data():
    def __init__(self):
        datapath = prefix + 'milldata/'
        x = np.array(pd.read_excel(datapath+'模拟数据.xlsx'))
        m = x.shape[0]
        m_length = int(m*0.7)
        x_train = x[:m_length]
        x_test = x[m_length:]
        self.train_data  = x_train
        self.test_data = x_test

class Synthetic_data():
    def __init__(self,sample_num,feather_num):
        U = np.random.normal(0,2*np.pi,sample_num)
        Z = np.random.normal(0,0.01,feather_num)        # variance 可以调整
        F = np.random.normal(0,2*np.pi,sample_num)
        X = 2*np.pi*np.linspace(0,feather_num-1,feather_num)/feather_num
        X = np.tile(X,sample_num).reshape(sample_num,feather_num)
        X = X.T+U
        F = np.tile(F,feather_num).reshape(feather_num,sample_num)
        S = np.sin(np.multiply(F,X))
        S = S.T+Z
        self.data = S




class Data_Transformer():
    def transform2D(self,data):
        m = data.shape[0]
        n = data.shape[1]
        data_2D_list = []
        number = int(m / n)
        for i in range(number):
            data_2D_line = data[i * n:i * n + n]
            data_2D_list.append(data_2D_line)
        return np.array(data_2D_list)

    def inverse_transform2D(self,data_2D):
        k = data_2D.shape[0]
        m = data_2D.shape[1]
        n = data_2D.shape[2]
        data_array = data_2D[0]
        for i in range(1,k):
            data_array = np.concatenate((data_array,data_2D[1]),axis=0)
        return  data_array


# nolinaer


# noGussian


# time_lag


# mutimode


# mutilevel


if __name__ == '__main__':
    # data = Synthetic_data(1000,20).data
    data = CSTR_data().data
    print(data)
