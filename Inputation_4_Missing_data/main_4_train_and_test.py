# written by LorneM 2019.12.1

import os,sys
sys.path.append('utils')
from datas import *
import numpy as np
import pandas as pd
from sklearn import preprocessing
from  Missing_data import missing_data
import copy
import random
from Medium_4_missingdata import Medium_4_missingdata
from Knn_4_missingdata import Knn_4_missingdata
# from AE_4_missingdata import AE_4_missingdata
# import h5py
# from keras.models import load_model


# 设置seed
random.seed(1)

#   ############## Load Data ##############
#   TE_data ——    TE正常工况数据，数据存放在文件d00.txt
#   CSTR_data ——    CSTR 正常工况数据
#   mill_data   ——    磨煤机数据
#   Synthetic_data  ——  仿真数据
mill_dataset = mill_data()
train_data = mill_dataset.train_data
test_data = mill_dataset.test_data
#   ############# Data Pre-processing ######################
# min_max_scaler = preprocessing.MinMaxScaler()   # 0-1 scale
# scaled_train_data = min_max_scaler.fit_transform(train_data)
# scaled_test_data = min_max_scaler.fit_transform(test_data)
normal_scaler = preprocessing.StandardScaler()  # standard scale
scaled_train_data = normal_scaler.fit_transform(train_data)
scaled_test_data = normal_scaler.fit_transform(test_data)



#   ############# Data Missing ######################
# normal_data_set —— original data set
# percent ——  missing percentage
# batch_size ——  batch_size for continuous missing
missing_test_data = copy.copy(scaled_test_data)
miss=missing_data()
precent = 0.95
batch_size =1
missing_test_data = miss.missing_data_for_matrix_batch(missing_test_data,precent,batch_size)



#   ################  Fill Data #########################
#   使用pandas缺失值补偿方法,归一化平均值为0
# missing_test_data_mean = copy.copy(missing_test_data)
# missing_matrix = np.isnan(missing_test_data_mean).astype(int)
# filled_test_data = np.nan_to_num(missing_test_data_mean)
# missing_matrix_error = (abs(filled_test_data - scaled_test_data) * missing_matrix)/np.sum(missing_matrix)
# missing_error_sum = np.sum(missing_matrix_error)
# print("均值补偿平均误差为："+str(missing_error_sum) )

#   Medium_4_missingdata 缺失数据由距离最近的前一个或者后一个平均值数据填补
missing_test_data_medium = copy.copy(missing_test_data)
medium_miss = Medium_4_missingdata()
filled_test_data = medium_miss.fill(missing_test_data_medium)
medium_miss.fill_estimator(filled_test_data,scaled_test_data)
print("前后值补偿平均误差为："+ str(medium_miss.missing_error_sum))

#   KNN_4_missingdata  缺失数据由距离最近的k个数据填补
missing_test_data_knn = copy.copy(missing_test_data)
knn_miss = Knn_4_missingdata()
filled_test_data = knn_miss.fill(missing_test_data_knn,scaled_train_data,k=1)
knn_miss.fill_estimator(filled_test_data,scaled_test_data)
print("KNN补偿平均误差为：" + str(knn_miss.missing_error_sum))

#   GRU+GAN  学习训练数据



































# # find the best parameter k
# missing_error_sum_list = []
# for num in range(1,20):
#     missing_test_data_knn = copy.copy(missing_test_data)
#     filled_dataset = knn_miss.fill(missing_test_data_knn, scaled_train_data, k=num)
#     knn_miss.fill_estimator(filled_dataset, scaled_test_data)
#     missing_error_sum_list.append(knn_miss.missing_error_sum)
#
#
#   ######################## Plot #################################
# import matplotlib.pyplot as plt
# # 解决中文显示问题
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# # 开始画图
# plt.title('KNN补全策略——k值变化关系图')
# plt.plot(np.linspace(1,15,14,endpoint=False),missing_error_sum_list, color='green')
# plt.legend()  # 显示图例
#
# plt.xlabel('k')
# plt.ylabel('missing_error_sum')
# plt.show()
#








#   AE_4_missingdata  训练数据训练一个AE生成模型进行补全
# 载入模型
# autoencoder = load_model('models/TEdata_AE_1.h5')
# ae_miss = AE_4_missingdata(autoencoder=autoencoder)
# filled_dataset = ae_miss.fill(missing_dataset,iter=5)
# ae_miss.fill_estimator(filled_dataset,scaled_data_test)
# print(ae_miss.missing_error_sum)




