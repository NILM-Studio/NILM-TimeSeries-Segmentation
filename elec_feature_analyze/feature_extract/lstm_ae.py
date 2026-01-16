import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Masking
from tensorflow.keras.optimizers import Adam

# ===================== 1. 核心超参数（新增特征选择参数） =====================
latent_dim = 8  # 提取的特征维度
epochs = 50
batch_size = 32
learning_rate = 0.001
selected_feature_idx = 0  # 关键：选择第几个特征作为输入（索引从0开始，比如选第1个特征填0，第2个填1）

# ===================== 2. 加载并预处理数据（适配不等长+单特征选择） =====================
# 加载数据：data.npy是填充后的统一长度数据，seq_len.npy是每个样本的真实长度
# data.shape = (n_samples, data_len, n_features)
# seq_len.shape = (n_samples,)
data = np.load("data.npy")
seq_len = np.load("seq_len.npy")

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

# ===================== 3. 构建带Masking的LSTM自编码器（适配不等长+单特征） =====================
# 输入层：适配单特征的时序形状 (timesteps, 1)
input_layer = Input(shape=(timesteps, n_features))

# 关键：Masking层（忽略填充值，默认mask_value=0，需和data.npy的填充值一致）
# 作用：让LSTM只处理样本真实长度内的有效数据，填充的0值不参与计算
masking_layer = Masking(mask_value=0.0)(input_layer)

# 编码器：LSTM编码 → 得到时序核心特征（基于Masking后的有效数据）
encoder_lstm = LSTM(32, activation='relu', return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(masking_layer)  # 输入改为masking_layer
latent_features = Dense(latent_dim, activation='relu')(state_h)

# 解码器：把特征重构回原始时序形状（单特征维度）
decoder_input = RepeatVector(timesteps)(latent_features)
decoder_lstm = LSTM(32, activation='relu', return_sequences=True)
decoder_outputs = decoder_lstm(decoder_input)
# TimeDistributed适配单特征输出，保证输出形状和输入一致 (timesteps, 1)
output_layer = TimeDistributed(Dense(n_features, activation='linear'))(decoder_outputs)

# 构建完整模型+单独编码器模型
lstm_autoencoder = Model(inputs=input_layer, outputs=output_layer)
lstm_encoder_model = Model(inputs=input_layer, outputs=latent_features)  # 编码器用于特征提取

# ===================== 4. 训练模型 =====================
lstm_autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
history = lstm_autoencoder.fit(
    X, X,  # 无监督，输入=标签（Masking层会忽略填充值的重构误差）
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_split=0.2
)

# ===================== 5. 提取时序特征（核心一步） =====================
X_lstm_extracted_features = lstm_encoder_model.predict(X)

# ===================== 结果输出 =====================
print(f"\n原始单特征时序数据形状: {X.shape}")  # 输出 (n_samples, timesteps, 1)
print(f"LSTM提取的特征形状: {X_lstm_extracted_features.shape}")  # 输出 (n_samples, latent_dim)