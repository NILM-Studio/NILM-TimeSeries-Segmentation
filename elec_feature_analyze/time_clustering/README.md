# 时间序列聚类分析

主要用于对时间序列数据进行聚类分析，主要结构如下：
- dbscan.py:  DBSCAN算法实现，可以完成从数据聚类到可视化分析的全流程
- cluster_result_analyze.py:  对聚类结果进行可视化分析
- timeseries_clustering.py:  包含聚类的通用方法，包括将数据转化为聚类方法的输入格式



## DeTSEC聚类算法
### 输入（Inputs）

#### 1. 命令行参数
运行该脚本时需要传入3个命令行参数，分别为：
- `dirName`：数据文件所在的目录路径（字符串）。
- `n_dims`：多变量时间序列的变量维度数（整数，即每个时间步的特征数量）。
- `n_clusters`：聚类的目标簇数（整数，通常与数据集中的类别数一致）。


#### 2. 数据文件（需放在`dirName`目录下）
- `data.npy`：  
  存储多变量时间序列数据的Numpy数组，形状为 `(nSamples, n_dims * max_length)`，其中：  
  - `nSamples` 是样本数量；  
  - `max_length` 是所有序列的最大时间步长；  
  - 每个样本的时间序列被展平存储（例如：若某样本有20个时间步，每个时间步含4个变量，则该样本在`data.npy`中占 `20*4=80` 列）。  

- `seq_length.npy`：  
  存储每个序列的有效长度（已乘以`n_dims`）的Numpy数组，形状为 `(nSamples,)`。  
  例如：若某样本的有效时间步长为20，且`n_dims=4`，则该样本在`seq_length.npy`中对应的值为 `20*4=80`，用于标识`data.npy`中该样本的有效数据范围（其余为填充值）。  


### 输出（Outputs）

#### 1. 生成的目录
- 脚本会创建一个名为 `{dirName最后一级目录}_detsec512` 的文件夹（例如：若`dirName`为`./data/ecg`，则目录名为`ecg_detsec512`），用于存储输出文件。


#### 2. 输出文件（保存到上述目录或当前工作目录）
- `detsec_features.npy`：  
  提取的特征向量（嵌入向量），形状为 `(nSamples, embedding_dim)`，其中`embedding_dim`由自编码器的隐藏层维度决定（代码中设为512）。该特征是通过自编码器的编码器部分对输入时间序列进行特征学习后的结果。  

- `detsec_clust_assignment.npy`：  
  聚类标签数组，形状为 `(nSamples,)`，存储每个样本通过KMeans聚类得到的簇分配结果（标签值为0到`n_clusters-1`之间的整数）。