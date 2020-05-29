# writen by @LorneM 2019.11.30
import numpy as np

# ############### missing data ############
# normal_data_set —— original data set
# percent ——  missing percentage
# batch_size ——  batch_size for continuous missing
class missing_data():
    def missing_data_for_vector(self,vector,percent):
        self.length = len(vector)
        index = np.linspace(0,self.length-1,self.length)
        np.random.shuffle(index)
        index_missing = index[:int(percent*self.length)]
        for i in range(len(index_missing)):
            vector[int(index_missing[i])] = None
        return vector

    def missing_data_for_matrix(self,matrix,percent):
        self.length = matrix.shape[0]*matrix.shape[1]
        vector = matrix.reshape((self.length,1))
        index = np.linspace(0,self.length-1,self.length)
        np.random.shuffle(index)
        index_missing = index[:int(percent*self.length)]
        for i in range(len(index_missing)):
            vector[int(index_missing[i])] = None
        matrix = vector.reshape((matrix.shape[0],matrix.shape[1]))
        return  matrix

    def missing_data_for_tensor(self,tensor,percent):
        self.length = tensor.shape[0]*tensor.shape[1]*tensor.shape[2]
        vector = tensor.reshape((self.length, 1))
        index = np.linspace(0,self.length-1,self.length)
        np.random.shuffle(index)
        index_missing = index[:int(percent*self.length)]
        for i in range(len(index_missing)):
            vector[int(index_missing[i])] = None
        tensor = vector.reshape((tensor.shape[0],tensor.shape[1],tensor.shape[2]))
        return  tensor

    def missing_data_for_vector_batch(self,vector,percent,batch_size):
        self.length = len(vector)
        index = np.linspace(0,self.length-1,self.length)
        np.random.shuffle(index)
        index_missing = index[:int(percent*self.length)]
        for i in range(len(index_missing)):
            for j in range(batch_size):
                try:
                    vector[int(index_missing[i]+j)] = None
                except:
                    vector[int(index_missing[i])] = None
        return  vector

    def missing_data_for_matrix_batch(self,matrix,percent,batch_size):
        np.random.seed(10)
        self.sample_length = matrix.shape[0]
        self.feather_length = matrix.shape[1]
        for feather_index in range(self.feather_length):
            index = np.linspace(0, self.sample_length - 1, self.sample_length)
            np.random.shuffle(index)
            index_missing = index[:int(percent * self.sample_length)]
            for i in range(len(index_missing)):
                for j in range(batch_size):
                    try:
                        matrix[int(index_missing[i] + j),feather_index] = None
                    except:
                        matrix[int(index_missing[i]),feather_index] = None
        return  matrix

    def missing_data_for_tensor_batch(self,tensor,percent,batch_size):
        self.tensor_length = tensor.shape[0]
        self.sample_length = tensor.shape[1]
        self.feather_length = tensor.shape[2]
        self.number = self.tensor_length * self.sample_length * self.feather_length
        vector = tensor.reshape(self.number,1,1)
        index = np.linspace(0, self.number - 1, self.number)
        np.random.shuffle(index)
        index_missing = index[:int(percent * self.number)]
        for i in range(len(index_missing)):
            index =int(index_missing[i])
            vector[index] = None
        tensor = vector.reshape(self.tensor_length,self.sample_length,self.feather_length)
        return tensor



if __name__ == '__main__':
    # tensor = np.ones((10,10,10))
    # vector =  np.ones(100)
    matrix = np.ones((10,10))
    miss=missing_data()
    vector = miss.missing_data_for_matrix_batch(matrix,0.1,6)
    print(vector)