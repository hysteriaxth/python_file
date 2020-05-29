# writen by @LorneM 2019.9.28
# algorithms is written from Reconstruction-based Contribution for Process Monitoring with KernelPrincipal Component Analysis
# Carlos F. Alcala , S. Joe Qin,2010


from scipy.spatial.distance import pdist, squareform
from scipy import exp
from numpy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.spatial.distance import cdist

# para
# train_data_scaled  归一化训练数据
# test_data_scaled  归一化测试数据
# gamma  核参数
# n_components 主元个数

#################  KPCA ######################

class KPCA_model():
    def __init__(self,gamma=0.1,components_percent = 0.9,alpha = 0.99):
        # 经验公式
        self.gamma = gamma
        self.components_percent = components_percent
        self.alpha = alpha

    def train(self,Xtrain):
        self.Xtrain = Xtrain
        self.K = self.__ComputeKernel__(self.Xtrain,self.Xtrain)
        # Center the kernel matrix.
        N = self.K.shape[0]
        one_n = np.ones((N, N)) / N
        Kc = self.K - one_n.dot(self.K) - self.K.dot(one_n) + one_n.dot(self.K).dot(one_n)
        # Obtaining eigenpairs from the centered kernel matrix
        # numpy.linalg.eigh returns them in sorted order
        eigvals, eigvecs = eigh(Kc / N)
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]
        eigvecs_norm = eigvecs / ((N * eigvals) ** 0.5)
        eigvecs_norm = np.where(np.isnan(eigvecs_norm), 0, eigvecs_norm)
        # Decision for n_components
        eigvals_percent = np.divide(eigvals, np.sum(eigvals))
        per = 0  # 特征值百分比
        n_components = 0  # 主元个数
        for precent in eigvals_percent:
            per += precent
            n_components = n_components + 1
            if per > self.components_percent:
                break
        #  Calculate control limit ########
        # eigvals_ = eigvals[n_components:]
        # g_SPE = np.sum(eigvals_ ** 2) / (len(eigvals) - 1) * np.sum(eigvals_)
        # h_SPE = np.sum(eigvals_) ** 2 / np.sum(eigvals_ ** 2)
        # self.SPE_limit = g_SPE * chi2.ppf(self.alpha, h_SPE)
        self.eigvecs_norm = eigvecs_norm
        self.n_components = n_components
        # Collect the top k eigenvectors (projected samples)
        self.P = eigvecs_norm[:,:n_components]
        self.P_ = eigvecs_norm[:,n_components:]
        self.eigvals_Matrix = np.diag(eigvals[:n_components])
        # Calculate Control Limit with KDE method
        t = Kc @ self.P
        SPE = np.sum((Kc @ self.eigvecs_norm) ** 2, axis=1) - np.sum(t ** 2, axis=1)
        SPE = np.sort(SPE)
        n = len(SPE)
        h = 1.06 * np.std(SPE) * n **(-0.2)
        px = []
        for i in range(n):
            px.append(np.sum(exp(-(SPE[i]-SPE)**2/h)))
        px = px/np.sum(px)
        percent_sum = 0
        for i in range(len(SPE)):
            percent_sum= percent_sum+px[i]
            if (percent_sum>0.83):
                self.SPE_limit = SPE[i]
                break


    def test(self,Xtest,PlotType = 1):
        kt = self.__ComputeKernel__(Xtest,self.Xtrain)
        N = kt.shape[0]
        M = kt.shape[1]
        one_n_t = np.ones((N, M)) / M
        one_n = np.ones((self.K.shape[0], self.K.shape[0])) / self.K.shape[0]
        kt_c = kt - one_n_t.dot(self.K) - kt.dot(one_n) + one_n_t.dot(self.K).dot(one_n)
        t = kt_c @ self.P
        self.SPE = np.sum((kt_c @ self.eigvecs_norm) ** 2, axis=1) - np.sum(t ** 2, axis=1)
        self.T2 = np.diag(t@np.linalg.inv(self.eigvals_Matrix)@t.T)
        self.__plotContribution__(PlotType)
        # self.fault_labels = self.__fault_isolation_RBC__(Xtest)

    def __plotContribution__(self,PlotType):
        if (PlotType==0):
            print("")
        elif(PlotType==1):
            plt.figure()
            plt.plot(range(len(self.SPE)), self.SPE)
            plt.axhline(y=self.SPE_limit, color='r', linestyle='-')
            plt.title("Contribution for SPE"+"——gamma:"+str(self.gamma))
            plt.xlabel("numbers")
            plt.ylabel("Contribution")
            plt.show()
        elif(PlotType==2):
            plt.figure()
            plt.plot(range(len(self.T2)), self.T2)
            # plt.axhline(y=self.T2_limit, color='r', linestyle='-')
            plt.title("Contribution for T2"+"——gamma:"+str(self.gamma))
            plt.xlabel("numbers")
            plt.ylabel("Contribution")
            plt.show()

    def __fault_isolation_CP__(self, Test_Data):
        kt = self.__ComputeKernel__(Test_Data, self.Xtrain)
        N = kt.shape[0]
        M = kt.shape[1]
        one_n_t = np.ones((N, M)) / M
        one_n = np.ones((self.K.shape[0], self.K.shape[0])) / self.K.shape[0]
        kt_c = kt - one_n_t.dot(self.K) - kt.dot(one_n) + one_n_t.dot(self.K).dot(one_n)
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
        kt = self.__ComputeKernel__(Test_Data, self.Xtrain)
        N = kt.shape[0]
        M = kt.shape[1]
        one_n_t = np.ones((N, M)) / M
        one_n = np.ones((self.K.shape[0], self.K.shape[0])) / self.K.shape[0]
        kt_c = kt - one_n_t.dot(self.K) - kt.dot(one_n) + one_n_t.dot(self.K).dot(one_n)

        ln = np.ones((1,N))/N
        lm = np.ones((1,M))/M
        C = self.P @ self.eigvals_Matrix @ self.P_.T
        F = np.eye(N)-np.ones((N,N))/N

        for col in range(M):
            kz = kt[:,col]
            kz_ = kt_c[:, col]
            v = np.zeros((M,1))     # fault direction
            v[col] = v[col]+1

    def __ComputeKernel__(self,X1,X2):
        dis = cdist(X1, X2, metric='euclidean') ** 2
        kt = exp(-self.gamma ** 2 * dis)
        return kt












