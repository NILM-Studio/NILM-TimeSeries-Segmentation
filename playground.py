a = 123

import numpy as np

# 读取.npy文件
data = np.load('elec_feature_analyze/time_clustering/cluster_data/washing_machine/bilstm_ae_attention_features_cleaned_power_64_dim.npy')

# 打印数据
print(data)
