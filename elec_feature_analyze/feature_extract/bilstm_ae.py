import os

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Masking, Bidirectional, Concatenate  # 新增Bidirectional和Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# 1. 查看TensorFlow版本
print("TensorFlow版本：", tf.__version__)

# 2. 核心验证：检查是否有可用的GPU设备
print("="*50)
print("是否检测到GPU：", tf.config.list_physical_devices('GPU'))
print("="*50)

# 3. 查看所有可用设备（CPU+GPU）
physical_devices = tf.config.list_physical_devices()
print("所有可用计算设备：", physical_devices)

# 4. 进阶验证：查看GPU详细信息
gpu_devices = tf.config.list_logical_devices('GPU')
print("GPU逻辑设备详情：", gpu_devices)

# ===================== 1. 核心超参数（新增特征选择参数） =====================
latent_dim = 64  # 提取的特征维度
epochs = 50
batch_size = 32
learning_rate = 0.001
patience = 5

# ===================== 2. 加载并预处理数据（适配不等长+单特征选择） =====================
data_file = "../time_clustering/cluster_data/washing_machine/data_fusion.npy"
seq_file = "../time_clustering/cluster_data/washing_machine/seq_length_fusion.npy"
result_dir = "../time_clustering/cluster_data/washing_machine"

columns_dict = {
    'power': 1,
    'cleaned_power': 2,
    'high_freq': 3,
    'low_freq': 4
}
column_name = 'low_freq'
EXTERNAL_TAG = f'_{latent_dim}_dim'
feature_file_name = f"lstm_ae_features_{column_name}{EXTERNAL_TAG}.npy"
data = np.load(data_file)
data = np.expand_dims(data[:, :, columns_dict[column_name]], axis=-1)
seq_len = np.load(seq_file).astype(np.int32)  # 明确指定整型

# 【核心】选择单个特征作为模型输入（保留三维结构，适配LSTM的[样本数, 时间步, 特征数]要求）
# 切片方式：[:, :, idx:idx+1] 保证输出仍为三维，避免变成二维
# data_single_feature = data[:, :, selected_feature_idx:selected_feature_idx+1]

# 提取数据维度（动态适配，无需硬编码）
n_samples = data.shape[0]
timesteps = data.shape[1]  # 填充后的统一时间步长度
n_features = data.shape[2]  # 选择后固定为1

# 打印数据信息，验证加载和特征选择正确性
print(f"数据加载完成 | 总样本数: {n_samples} | 填充后时间步: {timesteps} | 输入特征数: {n_features}")
print(f"样本真实长度范围: {np.min(seq_len)} ~ {np.max(seq_len)}")
print(f"选择的特征索引: {column_name} | 输入数据最终形状: {data.shape}")

# 赋值为模型输入
X = data

# ===================== 2.5 数据归一化（新增：解决NaN的关键） =====================
# 深度学习对数据量级敏感，原始数据高达3000+，不归一化极易导致梯度爆炸（NaN）
X_min = X.min()
X_max = X.max()
X = (X - X_min) / (X_max - X_min + 1e-7)
# 计算归一化后的 Mask 值（原始 0.0 对应的新值）
scaled_mask_value = (0.0 - X_min) / (X_max - X_min + 1e-7)
print(f"数据归一化完成 | 范围: {X.min():.2f} ~ {X.max():.2f} | 修正后的 Mask 值: {scaled_mask_value:.4f}")

# ===================== 3. 构建带Masking的BiLSTM自编码器（适配不等长+单特征） =====================
# 输入层：适配单特征的时序形状 (timesteps, 1)
input_layer = Input(shape=(timesteps, n_features))

# 关键：Masking层（忽略填充值，需和归一化后的填充值一致）
masking_layer = Masking(mask_value=scaled_mask_value)(input_layer)

# ---------------------- 核心修改：单向LSTM → 双向LSTM（BiLSTM） ----------------------
# 编码器：BiLSTM编码（双向LSTM，捕捉时序数据的前向+后向依赖，每个方向16单元，总参数量与原32单元单向一致）
encoder_bilstm = Bidirectional(LSTM(16, activation='tanh', return_state=True))
# BiLSTM返回值：output, forward_h, forward_c, backward_h, backward_c
encoder_outputs, f_h, f_c, b_h, b_c = encoder_bilstm(masking_layer)
# 合并双向LSTM的隐藏状态（前向h + 后向h），作为潜在特征的输入
combined_h = Concatenate(axis=-1)([f_h, b_h])
latent_features = Dense(latent_dim, activation='relu')(combined_h)

# 解码器（保持不变，单向LSTM足够重构时序）
decoder_input = RepeatVector(timesteps)(latent_features)
decoder_lstm = LSTM(32, activation='tanh', return_sequences=True)
decoder_outputs = decoder_lstm(decoder_input)
# TimeDistributed适配单特征输出
output_layer = TimeDistributed(Dense(n_features, activation='linear'))(decoder_outputs)

# 构建完整模型+单独编码器模型
lstm_autoencoder = Model(inputs=input_layer, outputs=output_layer)
lstm_encoder_model = Model(inputs=input_layer, outputs=latent_features)

# ===================== 4. 训练模型 =====================
# 添加 gradient clipping (clipnorm=1.0) 防止梯度爆炸
lstm_autoencoder.compile(optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0), loss='mse')

# 配置EarlyStopping回调
earliest_stop = EarlyStopping(
    monitor='val_loss',  # 监控验证集损失
    patience=patience,  # 3个epoch没有提升就停止
    mode='min',  # 损失越小越好
    restore_best_weights=True,  # 恢复最佳权重
    verbose=1  # 打印停止信息
)

history = lstm_autoencoder.fit(
    X, X,  # 无监督，输入=标签（Masking层会忽略填充值的重构误差）
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_split=0.2,
    callbacks=[earliest_stop]  # 添加EarlyStopping回调
)

# ===================== 5. 提取时序特征（核心一步） =====================
X_lstm_extracted_features = lstm_encoder_model.predict(X)

# ===================== 结果输出 =====================
print(f"\n原始单特征时序数据形状: {X.shape}")  # 输出 (n_samples, timesteps, 1)
print(f"BiLSTM提取的特征形状: {X_lstm_extracted_features.shape}")  # 输出 (n_samples, latent_dim)

# ===================== 保存结果 =====================
# 保存提取的LSTM特征到result目录
feature_output_path = os.path.join(result_dir, feature_file_name)  # 重命名为bilstm区分
np.save(feature_output_path, X_lstm_extracted_features)
print(f"结果已保存到: {feature_output_path}")
