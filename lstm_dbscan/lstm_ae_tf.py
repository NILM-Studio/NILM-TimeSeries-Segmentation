import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

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
selected_feature_idx = 0  # 关键：选择第几个特征作为输入（索引从0开始，比如选第1个特征填0，第2个填1）
do_visualize = True      # 是否生成并显示可视化图像

# ===================== 2. 加载并预处理数据（适配不等长+单特征选择） =====================
# 加载数据：data.npy是填充后的统一长度数据，seq_len.npy是每个样本的真实长度
# data.shape = (n_samples, data_len, n_features)
# seq_len.shape = (n_samples,)
data = np.load("./lstm_dbscan/data_washing_machine/data.npy")
seq_len = np.load("./lstm_dbscan/data_washing_machine/seq_length.npy")  

# 【核心】选择单个特征作为模型输入（保留三维结构，适配LSTM的[样本数, 时间步, 特征数]要求）
# 切片方式：[:, :, idx:idx+1] 保证输出仍为三维，避免变成二维
data_single_feature = data[:, :, selected_feature_idx:selected_feature_idx+1]

# 提取数据维度（动态适配，无需硬编码）
n_samples = data_single_feature.shape[0]
timesteps = data_single_feature.shape[1]  # 填充后的统一时间步长度
n_features = data_single_feature.shape[2]  # 选择后固定为1

# 打印数据信息，验证加载和特征选择正确性
print(f"数据加载完成 | 总样本数: {n_samples} | 填充后时间步: {timesteps} | 输入特征数: {n_features}")
print(f"样本真实长度范围: {np.min(seq_len)} ~ {np.max(seq_len)}")
print(f"选择的特征索引: {selected_feature_idx} | 输入数据最终形状: {data_single_feature.shape}")

# 赋值为模型输入
X = data_single_feature

# ===================== 2.5 数据归一化（新增：解决NaN的关键） =====================
# 深度学习对数据量级敏感，原始数据高达3000+，不归一化极易导致梯度爆炸（NaN）
X_min = X.min()
X_max = X.max()
X = (X - X_min) / (X_max - X_min + 1e-7)
# 计算归一化后的 Mask 值（原始 0.0 对应的新值）
scaled_mask_value = (0.0 - X_min) / (X_max - X_min + 1e-7)
print(f"数据归一化完成 | 范围: {X.min():.2f} ~ {X.max():.2f} | 修正后的 Mask 值: {scaled_mask_value:.4f}")

# ===================== 3. 构建带Masking的LSTM自编码器（适配不等长+单特征） =====================
# 输入层：适配单特征的时序形状 (timesteps, 1)
input_layer = Input(shape=(timesteps, n_features))

# 关键：Masking层（忽略填充值，需和归一化后的填充值一致）
masking_layer = Masking(mask_value=scaled_mask_value)(input_layer)

# 编码器：LSTM编码（建议使用 tanh 激活函数，比 relu 更稳定）
encoder_lstm = LSTM(32, activation='tanh', return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(masking_layer)
latent_features = Dense(latent_dim, activation='relu')(state_h)

# 解码器
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
    patience=5,  # 3个epoch没有提升就停止
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
print(f"LSTM提取的特征形状: {X_lstm_extracted_features.shape}")  # 输出 (n_samples, latent_dim)

# ===================== 保存结果 =====================
# 保存提取的LSTM特征到result目录
result_dir = "./lstm_dbscan/data_washing_machine/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
np.save(f"{result_dir}detsec_features.npy", X_lstm_extracted_features)
print(f"结果已保存到: {result_dir}detsec_features.npy")

# ===================== 6. 可视化：t-SNE 分布图 =====================
if do_visualize:
    print("\n正在生成 t-SNE 可视化...")
    
    # 对提取的特征进行 t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_lstm_extracted_features)
    
    plt.figure(figsize=(12, 6))
    
    # 子图 1: t-SNE 特征分布
    plt.subplot(1, 2, 1)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, c='blue', edgecolors='none', s=20)
    plt.title("t-SNE Visualization of LSTM Latent Features")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 子图 2: 原始数据抽样展示 (展示前10个样本)
    plt.subplot(1, 2, 2)
    for i in range(min(10, n_samples)):
        plt.plot(X[i, :seq_len[i], 0], alpha=0.7, label=f'Sample {i}')
    plt.title("Original Time Series Samples (Normalized)")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    # plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # 保存图像
    plot_path = f"{result_dir}tsne_distribution.png"
    plt.savefig(plot_path, dpi=300)
    print(f"可视化图像已保存到: {plot_path}")
    
    # 显示图像
    plt.show()
