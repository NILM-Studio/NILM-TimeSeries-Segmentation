# 使用全流程
## 零、前期准备

拉下Github上的两个仓库
```
git clone https://github.com/NILM-Studio/claspy.git
git clone https://github.com/NILM-Studio/NILM-TimeSeries-Segmentation.git
```
接下来会用到这两个仓库的代码，会用clasp和TSS来代称这两个仓库

## 一、数据前期预处理
准备NILM数据集数据，可以是csv文件，dat文件或者npy文件都可以，因为用电器很多时间是不活动的，所以第一步需要提取用电器中活动的数据

在clasp中运行semantic_segmentation.py脚本，提取用电器的活动数据
- 基本逻辑： 当用电器的功率超过阈值时，就认为用电器开始了一次活动，当用电器的功率连续n秒小于阈值时，就认为用电器结束了一次活动
- 输入: 用电器原始数据，格式为.dat、.csv、.npy
- 输出: 一个文件夹，文件夹中包含用电器每次活动的数据，格式为{self.appliance_name}_{start_str}_{end_str}_{duration}s.csv，表示每次活动的开始时间、结束时间和持续时间

可调参数:
- APPLIANCE_NAME: 用电器的名称，用于命名输出的文件夹
- POWER_THRESHOLD: 用电器的功率阈值，用于判断用电器是否开始或结束一次活动
- MIN_DURATION_SECONDS: 用电器的功率连续n秒小于阈值时，才认为用电器结束了一次活动
- CONTEXT_SECONDS: 用电器的活动数据，会包含用电器的前n秒和后n秒的数据，用于后续的特征提取


## 二、数据预处理


## 三、用电器特征提取与聚类

### 1.用电器特征提取
在TSS中`elec_feature_analyze\feature_extract`文件夹负责特征提取工作，可以支持如下几种特征提取方式：
- bilstm_ae.py: 用bilstm-autoencoder模型提取用电器的特征
- lstm_ae.py: 用lstm-autoencoder模型提取用电器的特征
- bilstm_ae_attantion.py: 用带注意力机制的bilstm-autoencoder模型提取用电器的特征
- timesfm_ae.py: 用timesfm-autoencoder模型提取用电器的特征
- timesfm_ae_pytorch.py: 用PyTorch实现的timesfm-autoencoder模型提取用电器的特征

目前可以使用的是bilstm_ae、lstm_ae和timesfm_ae_pytorch，其他的暂未调通

#### 1.1 bilstm_ae.py 说明
**功能描述:** 使用双向LSTM自编码器(BiLSTM-Autoencoder)模型从时序数据中提取用电器的时序特征。该模型能够捕捉时序数据的前后依赖关系，适用于处理不等长的时序序列。

**输入:**
- `data_low_freq.npy`: 预处理后的时序数据，形状为 [样本数, 时间步, 特征数]
- `seq_length.npy`: 每个样本的真实序列长度，用于处理不等长时序数据

**输出:**
- `bilstm_ae_features.npy`: 提取的BiLSTM特征，形状为 [样本数, latent_dim]

**流程:**
1. 加载时序数据和序列长度
2. 选择单个特征作为模型输入
3. 对数据进行归一化处理
4. 构建带Masking的BiLSTM自编码器模型
5. 训练模型并使用EarlyStopping防止过拟合
6. 提取时序特征并保存到文件

**可调参数:**
- `latent_dim`: 提取的特征维度，默认为64
- `epochs`: 训练轮数，默认为50
- `batch_size`: 批量大小，默认为32
- `learning_rate`: 学习率，默认为0.001
- `selected_feature_idx`: 选择第几个特征作为输入，默认为0
- `patience`: EarlyStopping的耐心值，默认为5


#### 1.2 lstm_ae.py 说明
**功能描述:** 使用LSTM自编码器(LSTM-Autoencoder)模型从时序数据中提取用电器的时序特征。该模型使用单向LSTM结构，相比BiLSTM参数更少，计算效率更高，适用于处理不等长的时序序列。

**输入:**
- `data_low_freq.npy`: 预处理后的时序数据，形状为 [样本数, 时间步, 特征数]
- `seq_length.npy`: 每个样本的真实序列长度，用于处理不等长时序数据

**输出:**
- `lstm_ae_features.npy`: 提取的LSTM特征，形状为 [样本数, latent_dim]

**流程:**
1. 加载时序数据和序列长度
2. 选择单个特征作为模型输入
3. 对数据进行归一化处理
4. 构建带Masking的LSTM自编码器模型
5. 训练模型并使用EarlyStopping防止过拟合
6. 提取时序特征并保存到文件

**可调参数:**
- `latent_dim`: 提取的特征维度，默认为64
- `epochs`: 训练轮数，默认为50
- `batch_size`: 批量大小，默认为32
- `learning_rate`: 学习率，默认为0.001
- `selected_feature_idx`: 选择第几个特征作为输入，默认为0
- `patience`: EarlyStopping的耐心值，默认为5


### 2.用电器特征聚类

#### 2.1 dbscan.py 详细说明
**功能描述:** 使用DBSCAN密度聚类算法对提取的用电器时序特征进行聚类分析。该脚本支持多种距离度量方式，能够自动识别噪声点，适用于发现任意形状的聚类簇。

**输入:**
- `data.npy`: 原始时序数据，形状为 [样本数, 时间步, 特征数]
- `bilstm_ae_features.npy`: 特征提取后的时序特征，形状为 [样本数, 特征维度]
- `seq_length.npy`: 每个样本的真实序列长度，主要在可视化等步骤上知道当前序列长度

**输出:**
- 指定输出目录`output_dir`为`'./cluster_data/dbscan_result/{appliance_name}/{eps}_{min_pts}_{EXTERN_TAG}/'`，不存在会自动新建，其中
  - `{appliance_name}`: 用电器名称
  - `{eps}`: DBSCAN邻域半径参数
  - `{min_pts}`: DBSCAN最小样本数参数
  - `{EXTERN_TAG}`: 额外标签，用于标识不同的实验配置

接下来的若干内容都会保存到这个文件夹中
- `cluster_labels.npy`: 聚类标签数组
- `evaluation_metrics.json`: 聚类评估指标（轮廓系数、Davies-Bouldin指数、Calinski-Harabasz指数）
- `tsne.png`: t-SNE可视化图像，展示特征空间中的聚类分布
- `cluster_stacked.png`: 每个簇的原生数据堆叠可视化图像，展示簇内数据分布
- `cluster_center.png`: 每个簇的DTW重心可视化图像，展示簇内数据的中心趋势
- 文件夹`cluster_{n}`: 第n个簇的子文件夹，包含前200个原始数据可视化图像
- `cluster_{n}.npy`: 第n个簇中各个数据在原始数据中的下标index，形状为 [簇内样本数]

**执行流程:**
1. **数据加载**: `load_data()`加载原始数据、特征数据和序列长度数据
2. **数据预处理**: `normalize_features()`对特征矩阵进行归一化处理，提取有效时间序列，可选zscore或minmax归一化
3. **距离矩阵计算**: `get_distance_matrix()`根据配置的距离度量方式（euclidean/dtw/fastdtw）计算样本间的距离矩阵
   - 支持距离矩阵缓存，避免重复计算
   - 对于DTW距离，使用tslearn库的并行计算优化
4. **DBSCAN聚类**: 基于预计算的距离矩阵执行DBSCAN聚类
   - eps: 邻域半径参数
   - min_samples: 最小样本数参数
5. **结果保存**: 调用 `cluster_result_save` 保存聚类结果到`output_dir`文件夹中
   - 保存聚类标签到 `cluster_labels.npy`
   - 每个簇新建一个名为`cluster_id`的子文件夹,将前200个原始数据可视化图像保存到该子文件夹中,
   - 保存聚类分析报告
6. **结果评估**: 调用 `evaluate_clustering` 评估聚类效果，并且将评估效果保存到`output_dir`文件夹中
   - 使用`cluster_result_quantification`评估聚类效果指标，包括轮廓系数、Davies-Bouldin指数、Calinski-Harabasz指数，并且保存到`evaluation_metrics.json`中
   - 使用`visualize_cluster_results`可视化聚类结果，包括如下内容:
     - 使用原生数据遍历绘制每个簇的DTW重心，保存为`cluster_center.png`
     - 使用各个簇的原生数据进行各堆叠可视化，保存为`cluster_stacked.png` 
     - 使用t-SNE对特征进行可视化，保存为`tsne.png`

**可调参数:**
- `BASE_DIR`: 数据基础目录路径，一般是在第二节语义分割模块中输出的结果目录
- `FEATURES_PATH`: 特征数据文件路径
- `CLUSTER_METHOD`: 聚类方法（'dbscan' 或 'kmeans'）
- `eps`: DBSCAN邻域半径，默认为0.1
- `min_pts`: DBSCAN最小样本数，默认为20
- `metric`: 距离度量方式，支持 'euclidean'、'dtw'、'fastdtw'
