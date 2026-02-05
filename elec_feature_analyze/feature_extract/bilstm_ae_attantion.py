import os

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, RepeatVector, TimeDistributed, Dense, Masking,
    Bidirectional, AdditiveAttention  # 无需Concatenate/GlobalPooling
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# 1. 查看TensorFlow版本
print("TensorFlow版本：", tf.__version__)

# 2. 核心验证：GPU检查
print("="*50)
print("是否检测到GPU：", tf.config.list_physical_devices('GPU'))
print("="*50)

# 3. 查看所有可用设备
physical_devices = tf.config.list_physical_devices()
print("所有可用计算设备：", physical_devices)

# 4. GPU详细信息
gpu_devices = tf.config.list_logical_devices('GPU')
print("GPU逻辑设备详情：", gpu_devices)

# ===================== 1. 核心超参数 =====================
latent_dim = 64  # 每个时间步的特征维度（非全局特征）
epochs = 50
batch_size = 32
learning_rate = 0.001
selected_feature_idx = 0
patience = 5

# ===================== 2. 加载并预处理数据 =====================
data_file = "../time_clustering/cluster_data/washing_machine_freq/data_low_freq.npy"
seq_file = "../time_clustering/cluster_data/washing_machine_freq/seq_length.npy"
result_dir = "../time_clustering/cluster_data/washing_machine_freq"

data = np.load(data_file)
seq_len = np.load(seq_file)

# 选择单个特征（保留3维结构：样本数×时间步×1）
data_single_feature = data[:, :, selected_feature_idx:selected_feature_idx+1]
n_samples = data_single_feature.shape[0]
timesteps = data_single_feature.shape[1]
n_features = data_single_feature.shape[2]

# 打印数据信息
print(f"数据加载完成 | 样本数: {n_samples} | 时间步: {timesteps} | 特征数: {n_features}")
print(f"样本真实长度: {np.min(seq_len)} ~ {np.max(seq_len)}")
print(f"输入数据形状: {data_single_feature.shape}")

# 赋值为模型输入
X = data_single_feature

# ===================== 2.5 数据归一化 =====================
X_min = X.min()
X_max = X.max()
X = (X - X_min) / (X_max - X_min + 1e-7)
scaled_mask_value = (0.0 - X_min) / (X_max - X_min + 1e-7)
print(f"归一化完成 | 范围: {X.min():.2f} ~ {X.max():.2f} | Mask值: {scaled_mask_value:.4f}")

# ===================== 3. 构建【纯时间步注意力】的BiLSTM自编码器 =====================
# 输入层：3维（None, T, 1）
input_layer = Input(shape=(timesteps, n_features), name="input_layer")

# Masking层（忽略填充值，适配不等长数据）
masking_layer = Masking(mask_value=scaled_mask_value, name="masking_layer")(input_layer)

# ========== 编码器：BiLSTM + 时间步注意力（全程保留3维） ==========
# 双向LSTM编码器（return_sequences=True → 输出3维序列：None,T,64）
encoder_bilstm = Bidirectional(
    LSTM(32, activation='tanh', return_sequences=True),  # 仅return_sequences=True，无需return_state
    name="encoder_bilstm"
)(masking_layer)

# 时间步注意力层（核心：保留3维输出，每个时间步都有注意力权重）
# 输入：(query=编码器序列, value=编码器序列) → 自注意力
attention_layer = AdditiveAttention(name="time_step_attention")
attention_output = attention_layer([encoder_bilstm, encoder_bilstm])  # 输出形状：(None, T, 64)

# 编码器特征降维（每个时间步降维到latent_dim=64，仍保留3维）
encoder_features = TimeDistributed(
    Dense(latent_dim, activation='relu'),
    name="encoder_time_step_dense"
)(attention_output)  # 输出形状：(None, T, 64) → 每个时间步64维特征

# ========== 解码器：基于3维时间步特征重构 ==========
# 解码器BiLSTM（return_sequences=True → 输出3维序列）
decoder_bilstm = Bidirectional(
    LSTM(32, activation='tanh', return_sequences=True),
    name="decoder_bilstm"
)(encoder_features)  # 直接输入3维编码器特征，无需RepeatVector

# 输出层：TimeDistributed适配每个时间步的重构
output_layer = TimeDistributed(
    Dense(n_features, activation='linear'),
    name="output_layer"
)(decoder_bilstm)  # 输出形状：(None, T, 1) → 和输入形状一致

# ========== 构建模型 ==========
# 完整自编码器（输入→输出，用于训练）
lstm_autoencoder = Model(inputs=input_layer, outputs=output_layer, name="bilstm_time_step_attention_ae")

# 编码器模型（输入→编码器特征，用于提取时间步注意力特征）
lstm_encoder_model = Model(inputs=input_layer, outputs=encoder_features, name="time_step_attention_encoder")

# ===================== 4. 训练模型 =====================
lstm_autoencoder.compile(
    optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
    loss='mse'
)

# 打印模型结构（重点看各层输出形状）
lstm_autoencoder.summary()

# 早停回调
earliest_stop = EarlyStopping(
    monitor='val_loss',
    patience=patience,
    mode='min',
    restore_best_weights=True,
    verbose=1
)

# 训练（输入=输出，无监督重构）
history = lstm_autoencoder.fit(
    X, X,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_split=0.2,
    callbacks=[earliest_stop]
)

# ===================== 5. 提取时间步注意力特征 =====================
# 提取的特征是3维：(n_samples, T, latent_dim) → 每个样本的每个时间步都有64维特征
X_time_step_features = lstm_encoder_model.predict(X)

# ===================== 结果输出 & 保存 =====================
print(f"\n原始数据形状: {X.shape}")
print(f"时间步注意力特征形状: {X_time_step_features.shape}")  # (n_samples, T, 64)

# 保存3维时间步特征
feature_output_path = os.path.join(result_dir, "time_step_attention_features.npy")
np.save(feature_output_path, X_time_step_features)
print(f"时间步注意力特征已保存到: {feature_output_path}")

# 可选：保存模型
model_output_path = os.path.join(result_dir, "time_step_attention_ae_model")
lstm_autoencoder.save(model_output_path)
print(f"模型已保存到: {model_output_path}")