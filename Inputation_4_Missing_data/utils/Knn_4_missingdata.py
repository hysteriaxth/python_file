# written by LorneM 2019.12.1
import numpy as np
import random
# 缺失数据由与未缺失部分最近的k个点加权和组成，权值未归一化后的距离倒数


class Knn_4_missingdata():


    def fill(self,test_data,train_data,k=4):
        missing_matrix = np.isnan(test_data).astype(int)
        unmissing_matrix = 1 - missing_matrix
        test_data = np.nan_to_num(test_data)
        self.missing_matrix =missing_matrix
        self.missing_num = np.sum(missing_matrix)
        for sample_index in range(test_data.shape[0]):
            sample = test_data[sample_index]
            missing_vector = missing_matrix[sample_index]
            if (np.sum(missing_vector)==0):
                continue
            unmissing_vector = unmissing_matrix[sample_index]
            unmissing_vector_2_matrix = np.tile(unmissing_vector,train_data.shape[0]).reshape((train_data.shape[0], train_data.shape[1]))
            sample_distance =np.sum((np.multiply(train_data,unmissing_vector_2_matrix)-sample)**2,axis=1)
            sample_distance = np.sqrt(sample_distance)
            sample_distance_sorted = np.sort(sample_distance)[:k]
            sample_distance_sorted_reciprocal = 1/(sample_distance_sorted+0.00000000001)
            sample_distance_sorted_reciprocal_scaled = sample_distance_sorted_reciprocal/np.sum(sample_distance_sorted_reciprocal)
            sample_distance_index = np.argsort(sample_distance)[:k]
            fill_matrix = train_data[sample_distance_index]
            fill_value =np.multiply(fill_matrix,np.tile(missing_vector,k).reshape((k, train_data.shape[1])))
            # fill_value_weighted = np.mean(fill_value,axis=0)    # 不用距离倒数作为权重
            fill_value_weighted =np.dot(sample_distance_sorted_reciprocal_scaled,fill_value)    # 用距离倒数作为权重
            test_data[sample_index] = test_data[sample_index] + fill_value_weighted
        self.fill_data = test_data
        return test_data

    def fill_estimator(self,data,original_data):
        self.missing_error_matrix = abs(data - original_data) * self.missing_matrix
        self.missing_error_sum = np.sum(self.missing_error_matrix) / self.missing_num


