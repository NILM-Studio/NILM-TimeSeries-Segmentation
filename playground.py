a = 123

import numpy as np

# 读取.npy文件
data = np.load('elec_feature_analyze/time_clustering/cluster_data/dbscan_result/washing_machine/0.6_20_bilistm/cluster_labels.npy')
data2 = np.load('elec_feature_analyze/time_clustering/cluster_data/dbscan_result/washing_machine_freq/0.1_20_low_freq_bilistm_ofd/cluster_result_0.1_20_low_freq_bilistm.npy')
# 打印数据
print(data)
