# written by LorneM 2019.12.1
import numpy as np
import random
# 缺失数据由距离最近的前一个和后一个的平均值数据填补,如果没有前一个或后一个，就用后一个或前一个单独代替


class Medium_4_missingdata():
    def fill(self,data):
        missing_matrix = np.isnan(data).astype(int)
        self.missing_matrix =missing_matrix
        self.missing_num = np.sum(missing_matrix)
        for feather_index in range(data.shape[1]):
            missing_vector = missing_matrix[:, feather_index]
            # 打乱补全顺序
            sample_index_list =np.linspace(0,data.shape[0],data.shape[0],endpoint=False).astype(int)
            # random.shuffle(sample_index_list)

            for sample_index in sample_index_list:
                number_flag = missing_vector[sample_index]
                if (not number_flag):
                    continue
                # 选择前面的数还是后面的数来补全
                pre_distance = 9999999999
                next_distance = 9999999999

                for pre_index in range(sample_index):
                    if (missing_vector[sample_index - pre_index] == False):
                        pre_distance = pre_index
                        break

                for next_index in range(data.shape[0] - sample_index):
                    if (missing_vector[sample_index + next_index] == False):
                        next_distance = next_index
                        break
                if(pre_distance<9999999999 and next_distance<9999999999):
                    data[sample_index, feather_index] = (data[sample_index - pre_distance, feather_index]+data[
                        sample_index + next_distance, feather_index])/2
                elif(pre_distance<9999999999):
                    data[sample_index, feather_index] = data[
                        sample_index - pre_distance, feather_index]
                else:
                    data[sample_index, feather_index] = data[
                        sample_index + next_distance, feather_index]
        self.fill_data = data
        return data

    def fill_estimator(self,data,original_data):
        self.missing_error_matrix = abs(data - original_data) * self.missing_matrix
        self.missing_error_sum = np.sum(self.missing_error_matrix) / self.missing_num


