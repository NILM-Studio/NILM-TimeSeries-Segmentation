a = 123

import numpy as np

# 读取.npy文件
data = np.load('elec_feature_analyze/time_clustering/cluster_data/dbscan_result/washing_machine/0.26_20_bilistm/Cluster_1.npy', allow_pickle=True)
# 打印数据
print(data)
