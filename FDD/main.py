# writen by @LorneM 2019.7.27  

import numpy as np
import pandas as pd
import csv
from Creat_Data import Normal_DataSet
from sklearn import preprocessing
from PCA import PCA_model
from KPCA import KPCA_model
import copy
import matplotlib.pyplot as plt
# import KPCA
# para
# train_data_scaled  归一化训练数据
# test_data_scaled  归一化测试数据
# gamma  核参数
# n_components 主元个数


#########################################################################################
# 读取工作簿和工作簿中的工作表 Read_data
# data_frame=pd.read_excel('DataSet\原数据\Test2015.xlsx',sheetname='Sheet1')
# Train_Data=np.array(data_frame)[1:]

# # 构造 线性数据
# Dataset_st = Normal_DataSet(2000,1000,10,1,1,5)
# DataSet_sample=Dataset_st.Creat_DataSet()
# train_data_array=DataSet_sample.Xtrain
# test_data_array=DataSet_sample.Fault_Data


# TE_data数据
# path='DataSet/TE_dataset/d06.csv'
# path1='DataSet/TE_dataset/d06_te.csv'
# train_data_array = np.array(pd.read_csv(path,header=None, dtype=float))[:,:52]
# test_data_array = np.array(pd.read_csv(path1,header=None, dtype=float))[:,:52]

# 燃气轮机excel数据
# train_data_df=pd.read_excel('Test2015.xlsx')
# train_data_array=np.array(train_data_df)[:,1:].astype(float)
# test_data_array=copy.copy(train_data_array)
# test_data_array[612:,10:12] =test_data_array[612:,10:12]

# 高加数据

# train_data_df=pd.read_excel('DataSet\High_Pressure_heater\data.xlsx',sheetname = 'Sheet1',header = None)
# time = np.array(train_data_df)[:,11]  # 时间
# train_data_array=np.array(train_data_df)[:102959,:11]
# test_data_array=np.array(train_data_df)[:,:11]
#
# inlet_pressure_3 = test_data_array[:,5]  # #3高加入口水压力
# inlet_steam_pressure_3 = test_data_array[:,8]  # #3高加入口蒸汽压力

#########################################################################################
# ################# 数据预处理 ############



scaler = preprocessing.StandardScaler()
train_data_scaled = scaler.fit_transform(train_data_array)
test_data_scaled =scaler.transform(test_data_array)
# test_data_scaled = copy.copy(train_data_scaled)
# test_data_scaled[1000:,14] = test_data_scaled[1000:,14] + 5
# m_test,n_test=np.shape(test_data_scaled)

#########################################################################################

# PCA
# PlotType = 1 denotes Plot for SPE , PlotType = 2 denotes Plot for T2
# type = 1 denotes one variable  Contribution Plot , type = 2 denotes one variable  Reconstruction Contribution Plot
model = PCA_model(components_percent=0.9)
model.train(train_data_scaled)
model.test(test_data_scaled, PlotType=1 ,type=2)
labels =np.array(model.fault_label)



# Rescon_Data = scaler.inverse_transform(model.Rescon_Data)
# inlet_pressure_3_Rescon_Data = Rescon_Data[:,5]   # #3高加入口水压力重构值
# inlet_steam_pressure_3_Rescon_Data = Rescon_Data[:,8]     # #3高加入口蒸汽压力重构值
#
# #   解决中文显示问题
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
#
# # 开始画图
# plt.figure()
# plt.title('#3高加入口水压力')
# plt.plot(time, inlet_pressure_3, color='green', label='实际值')
# plt.plot(time, inlet_pressure_3_Rescon_Data, color='red', label='重构值')
# plt.legend()  # 显示图例
# plt.xlabel('time')
# plt.ylabel('value')
# plt.show()
#
# # 开始画图
# plt.figure()
# plt.title('#3高加入口蒸汽压力')
# plt.plot(time, inlet_steam_pressure_3, color='green', label='实际值')
# plt.plot(time, inlet_steam_pressure_3_Rescon_Data, color='red', label='重构值')
# plt.legend()  # 显示图例
# plt.xlabel('time')
# plt.ylabel('value')
# plt.show()
#
# # 开始画图
# plt.figure()
# plt.title('高温加热器T2变化值')
# plt.plot(time, model.T2, color='black')
# plt.legend()  # 显示图例
# plt.xlabel('time')
# plt.ylabel('T2')
# plt.show()
#
# # 开始画图
# plt.figure()
# plt.title('高温加热器SPE变化值')
# plt.plot(time, model.SPE, color='red')
# plt.legend()  # 显示图例
# plt.xlabel('time')
# plt.ylabel('SPE')
# plt.show()

# KPCA

gamma = 0.1
model = KPCA_model()
model.train(train_data_scaled)
model.test(test_data_scaled, PlotType=1)


# gamma_list =np.linspace(0.01,0.13,12)
# for index in range(len(gamma_list)):
#     model = KPCA_model(gamma=gamma_list[index])
#     model.train(train_data_scaled)
#     model.test(test_data_scaled, PlotType=1)


# model estimation

# correct_rate=0
# for index in range(250,len(model.fault_label)):
#     if model.fault_label[index]==DataSet_sample.Fault_index[index]:
#         correct_rate=correct_rate+1
# correct_rate/=len(model.fault_label)-250
# print(correct_rate)

