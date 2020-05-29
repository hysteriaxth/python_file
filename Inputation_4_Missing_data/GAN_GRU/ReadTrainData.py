# writen by @LorneM 2020.1.2
import numpy as np
import pandas as pd
import copy
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os,sys
sys.path.append('../utils')
sys.path.append('utils')
from  Missing_data import missing_data

prefix = 'dataset/'


class readTrainData():
    """
    读取训练数据
    """

    def __init__(self,time_steps=6,train_test_percent = 0.7, batchSize = 64 ,batch_size=1):
        """
        :param time_steps: 时间步长
        :param train_test_percent: 训练测试样本百分比
        :param batchSize: 样本batch
        :param batch_size: 缺失batch，默认为1
        """
        self.time_steps = time_steps
        self.batchSize = batchSize
        self.batch_size = batch_size
        datapath = prefix + 'milldata/'
        x = np.array(pd.read_excel(datapath+'模拟数据.xlsx'))
        m = x.shape[0]
        m_length = int(m*train_test_percent)
        x = x[:m_length]


        # 归一化
        standard_scalar = StandardScaler()
        x = standard_scalar.fit_transform(x)
        self.standard_scalar = standard_scalar
        self.x = x

        #  数据转换
        self.__transform__()

    def __transform__(self):
        """
        :return:
        """

        x_list = []
        for i in range(self.x.shape[0] - self.time_steps):
            x_list.append(self.x[i:i + self.time_steps])
        x = np.array(x_list)
        self.x = x

        # 缺失处理
        x_ord = copy.copy(self.x)
        x_missing = copy.copy(self.x)
        # miss = missing_data()
        # missing_rate_list = [0]
        # for missing_rate in missing_rate_list:
        #     if missing_rate == 0:
        #         continue
        #     x_ord = np.concatenate((x_ord, self.x), 0)
        #     x_copy = copy.copy(self.x)
        #     x_missing_line = miss.missing_data_for_tensor_batch(x_copy, missing_rate, self.batch_size)
        #     x_missing = np.concatenate((x_missing, x_missing_line), 0)

        self.x_missing = x_missing

        # m
        m =1 - np.isnan(self.x_missing).astype(int)

        # detla
        detla = np.zeros((m.shape))

        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if j == 0:
                    continue
                for k in range(m.shape[2]):
                    detla[i][j][k] = 1 if m[i][j-1][k] == 1 else 1+detla[i][j-1][k]

        # imputed_deltapre
        imputed_deltapre = m

        self.x = x_ord
        self.x_missing = np.nan_to_num(x_missing)
        self.m = m
        self.detla = detla
        self.maxLength = self.x.shape[1]
        self.x_lengths = [self.time_steps]*x.shape[0]
        self.time = np.array([i for i in range(self.time_steps)]*x.shape[0]).reshape(x.shape[0],x.shape[1])

    def nextBatch(self):

        i=1
        while i * self.batchSize <= len(self.x):
            # x.append(self.x[(i-1)*self.batchSize:i*self.batchSize])
            # m.append(self.m[(i-1)*self.batchSize:i*self.batchSize])
            # detla.append(self.detla[(i-1)*self.batchSize:i*self.batchSize])
            # x_lengths.append(self.x_lengths[(i-1)*self.batchSize:i*self.batchSize])
            # time.append(self.time[(i-1)*self.batchSize:i*self.batchSize])
            x = self.x[(i-1)*self.batchSize:i*self.batchSize]
            x_missing = self.x_missing[(i - 1) * self.batchSize:i * self.batchSize]
            m = self.m[(i-1)*self.batchSize:i*self.batchSize]
            detla = self.detla[(i-1)*self.batchSize:i*self.batchSize]
            x_lengths = self.x_lengths[(i-1)*self.batchSize:i*self.batchSize]
            time = self.time[(i-1)*self.batchSize:i*self.batchSize]
            i = i+1
            yield x,x_missing, m, detla, x_lengths, time

    def shuffle(self, batchSize=64, isShuffle=False):
        self.batchSize = batchSize
        if isShuffle:
            c = list(
                zip(self.x, self.x_missing, self.m, self.detla , self.x_lengths , self.time
                    ))
            random.shuffle(c)
            self.x,self.x_missing, self.m, self.detla , self.x_lengths , self.time = zip(
                *c)



if __name__ == '__main__':

    print("success")
