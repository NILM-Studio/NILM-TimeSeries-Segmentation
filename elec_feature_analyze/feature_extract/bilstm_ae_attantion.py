import os

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, RepeatVector, TimeDistributed, Dense, Masking,
    Bidirectional, AdditiveAttention
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf


def bilstm_ae_attention(data: np.ndarray, config: dict):
    """
    BiLSTM + Attention 自编码器特征提取函数
    
    该函数使用双向 LSTM 结合注意力机制的自编码器从时序数据中提取时间步级别的特征。
    注意力机制能够自动学习每个时间步的重要性，对重要的时间步赋予更高的权重。
    
    与传统自编码器的区别：
    - 传统自编码器输出全局特征 (n_samples, latent_dim)，丢失了时间信息
    - 该方法输出时间步级别的特征 (n_samples, timesteps, latent_dim)，保留了时间结构
    
    模型结构：
    - 编码器：BiLSTM(32+32) + AdditiveAttention + TimeDistributed(Dense(latent_dim))
      - BiLSTM 提取双向时序特征，每个时间步都有 64 维特征
      - AdditiveAttention 学习时间步之间的依赖关系和重要性
      - TimeDistributed(Dense) 对每个时间步独立降维到 latent_dim
    - 解码器：BiLSTM(32+32) + TimeDistributed(Dense)，重构原始时序
    
    Args:
        data (np.ndarray): 输入数据，形状为 (n_samples, timesteps, n_features)
                          - n_samples: 样本数量
                          - timesteps: 时间步长度（填充后的统一长度）
                          - n_features: 特征数量（单特征时为1）
        config (dict): 模型配置字典，包含以下键：
                      - latent_dim (int): 每个时间步的特征维度，默认64
                      - epochs (int): 训练轮数，默认50
                      - batch_size (int): 批量大小，默认32
                      - learning_rate (float): 学习率，默认0.001
                      - patience (int): 早停耐心值，默认5
    
    Returns:
        np.ndarray: 提取的时间步注意力特征，形状为 (n_samples, timesteps, latent_dim)
                    - n_samples: 样本数量
                    - timesteps: 时间步长度
                    - latent_dim: 每个时间步的特征维度
    
    Example:
        >>> import numpy as np
        >>> data = np.random.rand(100, 50, 1)  # 100个样本，50个时间步，1个特征
        >>> config = {"latent_dim": 64, "epochs": 50, "batch_size": 32, 
        ...           "learning_rate": 0.001, "patience": 5}
        >>> features = bilstm_ae_attention(data, config)
        >>> print(features.shape)  # (100, 50, 64)
    
    Note:
        注意力机制的优势：
        - 自动学习每个时间步的重要性，无需人工设计特征
        - 能够捕捉长距离的时序依赖关系
        - 对噪声和异常值具有鲁棒性
        - 保留了完整的时间结构信息
        
        适用场景：
        - 需要分析时序数据中关键时间段的任务
        - 时序数据中存在重要事件或模式识别
        - 需要保留时间信息进行后续分析
    """
    # ===================== 1. 解析配置参数 =====================
    latent_dim = config["latent_dim"]  # 每个时间步的特征维度
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
    print(f"归一化完成 | 范围: {X.min():.2f} ~ {X.max():.2f} | Mask值: {scaled_mask_value:.4f}")

    # ===================== 4. 构建 BiLSTM + Attention 自编码器模型 =====================
    # 输入层：适配单特征的时序形状 (timesteps, n_features)
    input_layer = Input(shape=(timesteps, n_features), name="input_layer")

    # Masking 层：忽略填充值，适配不等长数据
    # 对于填充为 0.0 的位置，Masking 层会将其在计算中忽略
    masking_layer = Masking(mask_value=scaled_mask_value, name="masking_layer")(input_layer)

    # ===================== 编码器部分 =====================
    # BiLSTM 编码器：双向 LSTM，提取时序数据的前向+后向依赖
    # - 每个方向 32 个单元，总共 64 个单元
    # - return_sequences=True: 返回每个时间步的输出（保持 3 维结构）
    # - 输出形状: (batch_size, timesteps, 64)
    encoder_bilstm = Bidirectional(
        LSTM(32, activation='tanh', return_sequences=True),
        name="encoder_bilstm"
    )(masking_layer)

    # 注意力层：AdditiveAttention（也称为 Bahdanau Attention）
    # - 自动学习时间步之间的依赖关系和重要性
    # - 输入: [query, value]，这里使用自注意力（query=value=编码器输出）
    # - 输出形状: (batch_size, timesteps, 64)，与输入相同
    attention_layer = AdditiveAttention(name="time_step_attention")
    attention_output = attention_layer([encoder_bilstm, encoder_bilstm])

    # 编码器特征降维：对每个时间步独立应用全连接层
    # - TimeDistributed: 对每个时间步独立应用相同的 Dense 层
    # - 将每个时间步的 64 维特征降维到 latent_dim
    # - 输出形状: (batch_size, timesteps, latent_dim)
    encoder_features = TimeDistributed(
        Dense(latent_dim, activation='relu'),
        name="encoder_time_step_dense"
    )(attention_output)

    # ===================== 解码器部分 =====================
    # BiLSTM 解码器：从时间步特征重构时序数据
    # - 使用双向 LSTM（与编码器对称）
    # - return_sequences=True: 返回每个时间步的输出
    # - 输出形状: (batch_size, timesteps, 64)
    decoder_bilstm = Bidirectional(
        LSTM(32, activation='tanh', return_sequences=True),
        name="decoder_bilstm"
    )(encoder_features)

    # 输出层：TimeDistributed + Dense
    # - 对每个时间步独立应用全连接层
    # - 将解码器的输出映射到原始特征空间
    # - 输出形状: (batch_size, timesteps, n_features)
    output_layer = TimeDistributed(
        Dense(n_features, activation='linear'),
        name="output_layer"
    )(decoder_bilstm)

    # ===================== 5. 构建完整模型 =====================
    # 自编码器模型：用于训练，输入和输出都是原始数据
    # - 输入形状: (batch_size, timesteps, n_features)
    # - 输出形状: (batch_size, timesteps, n_features)
    lstm_autoencoder = Model(inputs=input_layer, outputs=output_layer, name="bilstm_time_step_attention_ae")
    
    # 编码器模型：用于特征提取，只保留编码器部分
    # - 输出形状: (batch_size, timesteps, latent_dim)
    # - 这是时间步级别的特征，保留了完整的时间结构
    lstm_encoder_model = Model(inputs=input_layer, outputs=encoder_features, name="time_step_attention_encoder")

    # ===================== 6. 编译模型 =====================
    # 使用 Adam 优化器，添加梯度裁剪防止梯度爆炸
    # clipnorm=1.0: 将梯度范数裁剪到 1.0 以内
    lstm_autoencoder.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss='mse'
    )

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
    # 输出形状: (n_samples, timesteps, latent_dim)
    # 这是时间步级别的特征，每个时间步都有 latent_dim 维的特征向量
    X_time_step_features = lstm_encoder_model.predict(X)

    # ===================== 10. 输出结果 =====================
    print(f"\n原始数据形状: {X.shape}")  # 输出 (n_samples, timesteps, n_features)
    print(f"时间步注意力特征形状: {X_time_step_features.shape}")  # 输出 (n_samples, timesteps, latent_dim)
    
    return X_time_step_features
