# writen by @LorneM 2019.9.28
# algorithms is written from Reconstruction-based contribution for process monitoring
# Carlos F. Alcala , S. Joe Qin,2010

import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.stats import norm
from Creat_Data import Normal_DataSet
import matplotlib.pyplot as plt
from scipy.stats import chi2
##################  读取数据  ############

class PCA_model():
    def __init__(self,components_percent = 0.9,alpha = 0.99):
        self.components_percent = components_percent
        self.alpha = alpha

    def train(self,Train_Data):
        covX = np.cov(Train_Data.T)  # 计算协方差矩阵
        featValue, featVec = np.linalg.eig(covX)  # 求解协方差矩阵的特征值和特征向量
        index = np.argsort(-featValue)  # 按照featValue进行从大到小排序
        featValue_percent = np.divide(featValue, np.sum(featValue))
        per = 0  # 特征值百分比
        n_components = 0  # 主元个数
        m, n = np.shape(Train_Data)
        for precent in featValue_percent:
            per += precent
            n_components = n_components + 1
            if per > self.components_percent:
                break
        if n_components > n:
            print("k must lower than feature number")
        else:
            self.selectVec = np.array(featVec.T[index[:n_components]])
            self.unselectVec = np.array(featVec.T[index[n_components:]])
            self.featValue = np.diag(featValue[index[:n_components]])
        # 计算阈值
        theta1 = np.sum(featValue[index[n_components:]])
        theta2 = np.sum(np.power(featValue[index[n_components:]], 2))
        theta3 = np.sum(np.power(featValue[index[n_components:]], 3))
        h0 = 1 - 2 * theta1 * theta3 / 3 * (np.power(theta2, 2))
        ca_99 = norm.ppf(self.alpha, loc=0, scale=1) # alpha = 0.99 by default
        self.SPE_limit = theta1 * np.power(
            h0 * ca_99 * np.sqrt(2 * theta2) / theta1 + 1 + theta2 * h0 * (h0 - 1) / np.power(theta1, 2),
            1 / h0)
        self.T2_limit = chi2.ppf(self.alpha,n_components, loc=0, scale=1)  # alpha = 0.99 by default



    # type=1 denotes one variable  Contribution Plot
    # type=2 denotes one variable  Reconstruction Contribution Plot
    def test(self,Test_Data,PlotType = 1,type=2):
        self.SPE = []
        self.T2 = []
        self.Rescon_Data = Test_Data@ self.selectVec.T@self.selectVec
        for i in range(Test_Data.shape[0]):
            SPE_line = Test_Data[i] @ self.unselectVec.T @ self.unselectVec @ Test_Data[i].T
            T2_line = Test_Data[i] @ self.selectVec.T @self.featValue @ self.selectVec @ Test_Data[i].T
            self.SPE.append(SPE_line)
            self.T2.append(T2_line)
        self.__plotContribution__(PlotType)
        if (type==1):
            self.fault_label = self.__fault_isolation_CP__(Test_Data)
        else:
            self.fault_label = self.__fault_isolation_RBC__(Test_Data)

    def __plotContribution__(self,PlotType):
        if (PlotType==0):
            print("")
        elif(PlotType==1):
            plt.figure()
            plt.plot(range(len(self.SPE)), self.SPE)
            plt.axhline(y=self.SPE_limit, color='r', linestyle='-')
            plt.title("Contribution for SPE")
            plt.xlabel("numbers")
            plt.ylabel("Contribution")
            plt.show()
        elif(PlotType==2):
            plt.figure()
            plt.plot(range(len(self.T2)), self.T2)
            plt.axhline(y=self.T2_limit, color='r', linestyle='-')
            plt.title("Contribution for T2")
            plt.xlabel("numbers")
            plt.ylabel("Contribution")
            plt.show()

        # one variable  Contribution Plot
        # number denotes fault index , and -1 denotes free fault

    def __fault_isolation_CP__(self, Test_Data):
        C_ = self.unselectVec.T @ self.unselectVec
        C_SPE = ((C_ @ Test_Data.T) ** 2).T
        labels = []
        for index in range(C_SPE.shape[0]):
            if (self.SPE[index] > self.SPE_limit):
                labels.append(np.where(C_SPE[index] == np.max(C_SPE[index], axis=0))[0][0])
            else:
                labels.append(-1)
        return labels

        # one variable  Reconstruction Contribution Plot

    def __fault_isolation_RBC__(self, Test_Data):
        C_ = self.unselectVec.T @ self.unselectVec
        C_SPE_RB = ((C_ @ Test_Data.T) ** 2).T / np.diag(C_)
        labels = []
        for index in range(C_SPE_RB.shape[0]):
            if (self.SPE[index] > self.SPE_limit):
                labels.append(np.where(C_SPE_RB[index] == np.max(C_SPE_RB[index], axis=0))[0][0])
            else:
                labels.append(-1)
        return labels

