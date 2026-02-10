import os

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Masking, Bidirectional, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf


def bilstm_ae(data: np.ndarray, config: dict):
    """
    BiLSTM 自编码器特征提取函数
    
    该函数使用双向 LSTM（BiLSTM）自编码器从时序数据中提取全局潜空间特征。
    BiLSTM 能够同时捕捉时序数据的前向和后向依赖关系，相比单向 LSTM 能更好地理解时序上下文。
    
    模型结构：
    - 编码器：BiLSTM(16+16) + Concatenate + Dense(latent_dim)，提取时序的全局特征
      - BiLSTM 每个方向 16 个单元，总共 32 个单元（与单向 LSTM 32 单元参数量相当）
      - 合并前向和后向的隐藏状态，捕捉双向时序依赖
    - 解码器：RepeatVector + LSTM(32) + TimeDistributed(Dense)，重构原始时序
    
    Args:
        data (np.ndarray): 输入数据，形状为 (n_samples, timesteps, n_features)
                          - n_samples: 样本数量
                          - timesteps: 时间步长度（填充后的统一长度）
                          - n_features: 特征数量（单特征时为1）
        config (dict): 模型配置字典，包含以下键：
                      - latent_dim (int): 提取的特征维度，默认64
                      - epochs (int): 训练轮数，默认50
                      - batch_size (int): 批量大小，默认32
                      - learning_rate (float): 学习率，默认0.001
                      - patience (int): 早停耐心值，默认5
    
    Returns:
        np.ndarray: 提取的特征，形状为 (n_samples, latent_dim)
                    - n_samples: 样本数量
                    - latent_dim: 特征维度（潜空间维度）
    
    Example:
        >>> import numpy as np
        >>> data = np.random.rand(100, 50, 1)  # 100个样本，50个时间步，1个特征
        >>> config = {"latent_dim": 64, "epochs": 50, "batch_size": 32, 
        ...           "learning_rate": 0.001, "patience": 5}
        >>> features = bilstm_ae(data, config)
        >>> print(features.shape)  # (100, 64)
    
    Note:
        BiLSTM 相比 LSTM 的优势：
        - 能够同时考虑过去和未来的信息
        - 对于需要理解完整时序上下文的任务效果更好
        - 参数量与单向 LSTM 相当（16+16 vs 32），但表达能力更强
    """
    # ===================== 1. 解析配置参数 =====================
    latent_dim = config["latent_dim"]  # 潜空间特征维度
    epochs = config["epochs"]          # 训练轮数
    batch_size = config["batch_size"]  # 批量大小
    learning_rate = config["learning_rate"]  # 学习率
    patience = config["patience"]      # 早停耐心值

    # ===================== 2. 提取数据维度信息 =====================
    timesteps = data.shape[1]  # 时间步长度
    n_features = data.shape[2]  # 特征数量
    
    # 赋值为模型输入
    X = data

    # ===================== 3. 数据归一化 =====================
    # 深度学习对数据量级敏感，原始数据可能高达数千，不归一化极易导致梯度爆炸（NaN）
    # 使用 Min-Max 归一化将数据映射到 [0, 1] 区间
    X_min = X.min()
    X_max = X.max()
    X = (X - X_min) / (X_max - X_min + 1e-7)  # 加上小常数防止除零
    
    # 计算归一化后的 Mask 值（原始填充值 0.0 对应的新值）
    # Masking 层会忽略这个值，避免填充值对模型训练的影响
    scaled_mask_value = (0.0 - X_min) / (X_max - X_min + 1e-7)
    print(f"数据归一化完成 | 范围: {X.min():.2f} ~ {X.max():.2f} | 修正后的 Mask 值: {scaled_mask_value:.4f}")

    # ===================== 4. 构建 BiLSTM 自编码器模型 =====================
    # 输入层：适配单特征的时序形状 (timesteps, n_features)
    input_layer = Input(shape=(timesteps, n_features))

    # Masking 层：忽略填充值，适配不等长数据
    # 对于填充为 0.0 的位置，Masking 层会将其在计算中忽略
    masking_layer = Masking(mask_value=scaled_mask_value)(input_layer)

    # ===================== 编码器部分 =====================
    # BiLSTM 编码器：双向 LSTM，捕捉时序数据的前向+后向依赖
    # - 每个方向 16 个单元，总共 32 个单元（与单向 LSTM 32 单元参数量相当）
    # - return_state=True: 返回前向和后向的隐藏状态和细胞状态
    # - activation='tanh': 使用 tanh 激活函数，比 relu 更稳定
    encoder_bilstm = Bidirectional(LSTM(16, activation='tanh', return_state=True))
    
    # BiLSTM 返回值：output, forward_h, forward_c, backward_h, backward_c
    # - output: 双向 LSTM 的输出序列
    # - forward_h: 前向 LSTM 的最终隐藏状态
    # - forward_c: 前向 LSTM 的最终细胞状态
    # - backward_h: 后向 LSTM 的最终隐藏状态
    # - backward_c: 后向 LSTM 的最终细胞状态
    encoder_outputs, f_h, f_c, b_h, b_c = encoder_bilstm(masking_layer)
    
    # 合并双向 LSTM 的隐藏状态（前向 h + 后向 h）
    # Concatenate 在最后一个维度拼接，形状从 (batch_size, 16) 变为 (batch_size, 32)
    combined_h = Concatenate(axis=-1)([f_h, b_h])
    
    # 全连接层：将合并后的隐藏状态映射到潜空间
    # combined_h 的形状为 (batch_size, 32)，经过 Dense 层后变为 (batch_size, latent_dim)
    latent_features = Dense(latent_dim, activation='relu')(combined_h)

    # ===================== 解码器部分 =====================
    # RepeatVector: 将潜空间特征重复 timesteps 次，生成解码器的输入序列
    # 输入形状: (batch_size, latent_dim) -> 输出形状: (batch_size, timesteps, latent_dim)
    decoder_input = RepeatVector(timesteps)(latent_features)
    
    # LSTM 解码器：从潜空间特征重构时序数据
    # - 使用单向 LSTM（单向足以重构，编码器已经提取了双向特征）
    # - return_sequences=True: 返回每个时间步的输出
    decoder_lstm = LSTM(32, activation='tanh', return_sequences=True)
    decoder_outputs = decoder_lstm(decoder_input)
    
    # TimeDistributed + Dense: 对每个时间步独立应用全连接层
    # 将解码器的输出映射到原始特征空间
    output_layer = TimeDistributed(Dense(n_features, activation='linear'))(decoder_outputs)

    # ===================== 5. 构建完整模型 =====================
    # 自编码器模型：用于训练，输入和输出都是原始数据
    lstm_autoencoder = Model(inputs=input_layer, outputs=output_layer)
    
    # 编码器模型：用于特征提取，只保留编码器部分
    lstm_encoder_model = Model(inputs=input_layer, outputs=latent_features)

    # ===================== 6. 编译模型 =====================
    # 使用 Adam 优化器，添加梯度裁剪防止梯度爆炸
    # clipnorm=1.0: 将梯度范数裁剪到 1.0 以内
    lstm_autoencoder.compile(optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0), loss='mse')

    # ===================== 7. 配置早停回调 =====================
    # 当验证集损失在 patience 个 epoch 内没有改善时，停止训练
    earliest_stop = EarlyStopping(
        monitor='val_loss',      # 监控验证集损失
        patience=patience,       # 耐心值：多少个 epoch 没有改善就停止
        mode='min',              # 损失越小越好
        restore_best_weights=True,  # 恢复到最佳权重的模型
        verbose=1                # 打印停止信息
    )

    # ===================== 8. 训练模型 =====================
    # 无监督学习：输入和标签都是原始数据
    # Masking 层会忽略填充值的重构误差
    history = lstm_autoencoder.fit(
        X, X,                    # 输入 = 标签（自编码器的特点）
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,             # 每个 epoch 打乱数据顺序
        validation_split=0.2,    # 20% 数据作为验证集
        callbacks=[earliest_stop] # 使用早停回调
    )

    # ===================== 9. 提取特征 =====================
    # 使用训练好的编码器提取特征
    X_lstm_extracted_features = lstm_encoder_model.predict(X)

    # ===================== 10. 输出结果 =====================
    print(f"\n原始单特征时序数据形状: {X.shape}")  # 输出 (n_samples, timesteps, n_features)
    print(f"BiLSTM提取的特征形状: {X_lstm_extracted_features.shape}")  # 输出 (n_samples, latent_dim)
    
    return X_lstm_extracted_features
