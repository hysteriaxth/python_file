#####  writen by @LorneM 2019.9.28##########


from scipy.stats import norm
import numpy as np
import itertools
import random
class Normal_DataSet:
    sample_num_train=0
    sample_num_test=0
    fault_num=0
    upperB=0
    lowerB=0
    feather_num=0
    DataSet = []
    def __init__(self,m_train,m_test,n,f_num,a,b):
        self.sample_num_train = m_train
        self.sample_num_test = m_test
        self.feather_num = n
        self.fault_num = f_num
        self.upperB = b
        self.lowerB = a
        self.Xtrain =[]
        self.Fault_index=[]
        self.Fault_Data=[]

    def Creat_DataSet(self):
        noise1= np.random.rand(self.sample_num_test, self.fault_num)*(self.upperB-self.lowerB)+self.lowerB
        noise1[:int(self.sample_num_test*0.25)]=0
        Fault_free=self.__Creat_NormalDataSet__()
        X_train = Fault_free[:self.sample_num_train]
        Fault_index = list(itertools.combinations(range(self.feather_num), self.fault_num))
        Fault_index=[val for val in Fault_index for i in range(20)]
        Fault_index = Fault_index[:self.sample_num_test]
        random.shuffle(Fault_index)
        Fault_index=np.array(Fault_index)
        Fault_Data = Fault_free[self.sample_num_train:]
        for i in range(len(Fault_index)):
            Fault_Data[i,Fault_index[i]]=Fault_Data[i,Fault_index[i]]+noise1[i]
        Fault_index[:int(self.sample_num_test*0.25)] = -1
        self.Xtrain=X_train
        self.Fault_index=Fault_index
        self.Fault_Data=Fault_Data
        return self

    def __Creat_NormalDataSet__(self):
        t1 = np.random.rand(self.sample_num_train+self.sample_num_test, 1)
        t2 = np.random.rand(self.sample_num_train+self.sample_num_test, 1)*0.8
        t3 = np.random.rand(self.sample_num_train+self.sample_num_test, 1)*0.6
        P = 2 * np.random.rand(self.feather_num,3)-1
        noise = 0.01 * np.random.rand(self.sample_num_train+self.sample_num_test,self.feather_num)
        X_train =P@np.row_stack((t1.T,t2.T,t3.T))+noise.T
        return X_train.T