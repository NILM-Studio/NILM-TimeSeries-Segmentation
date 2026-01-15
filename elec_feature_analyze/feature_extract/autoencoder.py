import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam

# ===================== 1. 时序数据（标准格式，无需展平！） =====================
n_samples = 500
timesteps = 30
n_features = 2
X = np.random.randn(n_samples, timesteps, n_features)  # 原始格式直接用

# ===================== 2. 超参数 =====================
latent_dim = 8  # 提取的特征维度
epochs = 50
batch_size = 32
learning_rate = 0.001

# ===================== 3. 构建LSTM自编码器（时序专用最优结构） =====================
# 【核心】编码器：LSTM编码 → 得到时序核心特征
# return_state=True：LSTM输出「输出值+隐藏状态+细胞状态」，隐藏状态是核心特征
input_layer = Input(shape=(timesteps, n_features))
# 编码：降维学习时序特征
encoder_lstm = LSTM(32, activation='relu', return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(input_layer)
# LSTM的隐含特征 = 隐藏状态h + 细胞状态c，拼接后得到最终特征
latent_features = Dense(latent_dim, activation='relu')(state_h)

# 解码器：把特征重构回原始时序形状
# RepeatVector：把特征向量复制timesteps次，适配时序长度
decoder_input = RepeatVector(timesteps)(latent_features)
decoder_lstm = LSTM(32, activation='relu', return_sequences=True)
decoder_outputs = decoder_lstm(decoder_input)
# TimeDistributed：对每个时间步做全连接，保证输出形状和输入一致
output_layer = TimeDistributed(Dense(n_features, activation='linear'))(decoder_outputs)

# 构建完整模型+单独编码器模型
lstm_autoencoder = Model(inputs=input_layer, outputs=output_layer)
lstm_encoder_model = Model(inputs=input_layer, outputs=latent_features)  # 编码器用于特征提取

# ===================== 4. 训练模型 =====================
lstm_autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
history = lstm_autoencoder.fit(
    X, X,  # 无监督，输入=标签
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_split=0.2
)

# ===================== 5. 提取时序特征（核心一步） =====================
X_lstm_extracted_features = lstm_encoder_model.predict(X)

# ===================== 结果输出 =====================
print(f"原始时序数据形状: {X.shape}")
print(f"LSTM提取的特征形状: {X_lstm_extracted_features.shape}")
# 输出：(500, 30, 2) → (500, 8)