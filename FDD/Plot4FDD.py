
import numpy as np
import pandas as pd
import csv
from Creat_Data import Normal_DataSet
from sklearn import preprocessing
from PCA import PCA_model
from KPCA import KPCA_model
import copy
import matplotlib.pyplot as plt


# 高加数据

train_data_df=pd.read_excel('DataSet\High_Pressure_heater\data.xlsx',sheetname = 'Sheet1',header = None)
train_data_array=np.array(train_data_df)

# ################# 数据预处理 ############
time = train_data_array[:,11]  # 时间
train_data_array = train_data_array[:,:11]
train_data_array = preprocessing.scale(train_data_array)

load = train_data_array[:,0]  # 负荷
pumpA_flow_3 = train_data_array[:,1]  # 汽泵A给水流量
outlet_temperature_3 = train_data_array[:,2]  # #汽泵B给水流量
inlet_temperature_3 = train_data_array[:,3]  # #3高加入口水温
FW_flow_3 = train_data_array[:,4]  # #锅炉给水流量
inlet_pressure_3 = train_data_array[:,5]  # #3高加入口水压力
outlet_temperature_3 = train_data_array[:,6]  # #3高加出口水温
water_level_3 = train_data_array[:,7]   # #3高加水位
inlet_steam_pressure_3 = train_data_array[:,8]  # #3高加入口蒸汽压力
inlet_steam_temperature_3 = train_data_array[:,9] #  #3高加入口蒸汽温度
dewatering_temperature_3 = train_data_array[:,10]  # #3高加疏水温度








#解决中文显示问题
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False


# 开始画图
# plt.title('高热加热器相关参数归一化后时序图')
# plt.plot(time, load, color='green', label='负荷')
# plt.plot(time, pumpA_flow_3, color='red', label='汽泵A给水流量')
# plt.plot(time, outlet_temperature_3, color='skyblue', label='汽泵B给水流量')
# plt.plot(time, inlet_temperature_3, color='blue', label='#3高加入口水温')
# plt.plot(time, FW_flow_3, color='darkred', label='锅炉给水流量')
# plt.plot(time, inlet_pressure_3, color='deeppink', label='#3高加入口水压力')
# plt.plot(time, outlet_temperature_3, color='forestgreen', label='#3高加出口水温')
# plt.plot(time, water_level_3, color='olive', label='#3高加水位')
# plt.plot(time, inlet_steam_pressure_3, color='rosybrown', label='#3高加入口蒸汽压力')
# plt.plot(time, inlet_steam_temperature_3, color='tan', label='#3高加入口蒸汽温度')
# plt.plot(time, dewatering_temperature_3, color='wheat', label='#3高加疏水温度')
# plt.legend()  # 显示图例
#
# plt.xlabel('time')
# plt.ylabel('value')
# plt.show()
# python 一个折线图绘制多个曲线



