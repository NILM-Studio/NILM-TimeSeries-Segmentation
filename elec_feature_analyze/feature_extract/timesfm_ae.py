# ===================== 1. 安装依赖（首次运行执行） =====================
# !pip install numpy tensorflow scikit-learn pywt tf-timesfm

# ===================== 2. 导入库 =====================
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, RepeatVector, TimeDistributed, Dense, Masking
from tensorflow.keras.optimizers import Adam
# 核心：导入谷歌官方TimesFM层（轻量化实现，适配Keras）
from tf_timesfm import TimeSeriesFourierLayer as TimesFM
# 聚类相关库（你的原有代码，完全不变）
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ===================== 3. 超参数设置（你的原有参数，完全不变，按需调整） =====================
latent_dim = 8        # 最终提取的特征维度
epochs = 50           # 训练轮数
batch_size = 32       # 批次大小
learning_rate = 0.001 # 学习率
selected_feature_idx = 0 # 选择第0个特征作为输入（你可修改为任意特征索引）
n_clusters = 3        # 聚类数量

# ===================== 4. 加载你的数据（完全不变，无缝对接） =====================
# data.npy: (n_samples, data_len, n_feature) 填充后的不等长时序数据
# seq_len.npy: (n_samples,) 每个样本的真实长度
data = np.load("data.npy")
seq_len = np.load("seq_len.npy")

# 选择单个特征，保留三维结构 (n_samples, data_len, 1) 【你的核心需求】
data_single_feature = data[:, :, selected_feature_idx:selected_feature_idx+1]
X = data_single_feature

# 提取数据维度（动态适配，无需硬编码）
n_samples, timesteps, n_features = X.shape
print(f"数据加载完成 | 样本数: {n_samples} | 时间步: {timesteps} | 特征数: {n_features}")
print(f"样本真实长度范围: {np.min(seq_len)} ~ {np.max(seq_len)}")

# ===================== 5. 构建TimesFM自编码器（核心修改：替换LSTM为TimesFM） =====================
input_layer = Input(shape=(timesteps, n_features))
# 关键层：Masking层，适配不等长数据，忽略填充的0值（完全不变）
masking_layer = Masking(mask_value=0.0)(input_layer)

# ✅ 编码器：TimesFM层 替代 LSTM层，自动高低频+时域建模，特征提取核心
timesfm_encoder = TimesFM(filters=32, kernel_size=3, activation='relu')
encoder_outputs = timesfm_encoder(masking_layer)
# 全局池化得到样本级特征（替代LSTM的state_h，完全不变）
encoder_pool = np.mean(encoder_outputs, axis=1)
# 降维得到最终核心特征
latent_features = Dense(latent_dim, activation='relu')(encoder_pool)

# ✅ 解码器：TimesFM层 替代 LSTM层，重构原始时序数据
decoder_input = RepeatVector(timesteps)(latent_features)
timesfm_decoder = TimesFM(filters=32, kernel_size=3, activation='relu', return_sequences=True)
decoder_outputs = timesfm_decoder(decoder_input)
# 输出层：和输入特征数一致，线性激活（重构任务）
output_layer = TimeDistributed(Dense(n_features, activation='linear'))(decoder_outputs)

# ===================== 6. 模型构建+训练（完全不变） =====================
# 完整自编码器：用于训练
timesfm_autoencoder = Model(inputs=input_layer, outputs=output_layer)
# 单独编码器：用于特征提取（核心产物）
timesfm_encoder_model = Model(inputs=input_layer, outputs=latent_features)

# 编译+训练：无监督，输入=标签，MSE损失
timesfm_autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
history = timesfm_autoencoder.fit(
    X, X,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_split=0.2
)

# ===================== 7. 提取低维特征（完全不变，格式一致） =====================
X_timesfm_features = timesfm_encoder_model.predict(X)
print(f"\nTimesFM提取的特征形状: {X_timesfm_features.shape}") # (n_samples, latent_dim)

# ===================== 8. 无监督聚类（你的原有代码，完全不变，无缝衔接） =====================
# 特征归一化（聚类必需）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_timesfm_features)

# KMeans聚类
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# 聚类效果评估（轮廓系数）
sil_score = silhouette_score(X_scaled, cluster_labels)
print(f"\nTimesFM特征聚类轮廓系数: {sil_score:.4f}")

# 输出每个聚类的样本数量
unique_labels, counts = np.unique(cluster_labels, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"聚类 {label} 的样本数: {count}")

# ===================== 9. 保存结果（可选） =====================
np.save("timesfm_extracted_features.npy", X_timesfm_features)
np.save("cluster_labels.npy", cluster_labels)