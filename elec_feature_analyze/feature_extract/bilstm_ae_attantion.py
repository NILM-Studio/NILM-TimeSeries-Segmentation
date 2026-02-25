import os

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, RepeatVector, TimeDistributed, Dense, Masking,
    Bidirectional, Permute, Multiply, Lambda, Layer
)
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf


def bilstm_ae_attention(data: np.ndarray, config: dict):
    """
    BiLSTM + 全局注意力自编码器特征提取函数
    
    该函数使用双向 LSTM 结合全局注意力机制的自编码器从时序数据中提取全局特征。
    全局注意力机制能够自动学习每个时间步的重要性，对重要的时间步赋予更高的权重，
    并将所有时间步的特征聚合为一个全局特征向量。
    
    与时间步级别注意力的区别：
    - 时间步级别：输出 (n_samples, timesteps, latent_dim)，保留时间信息
    - 全局级别：输出 (n_samples, latent_dim)，聚合时间信息，适合聚类等任务
    
    模型结构：
    - 编码器：BiLSTM(32+32) + 全局注意力 + Dense(latent_dim)
      - BiLSTM 提取双向时序特征，每个时间步都有 64 维特征
      - 全局注意力通过计算时间步权重并加权求和，聚合为全局特征
      - Dense 层将 64 维全局特征降维到 latent_dim
    - 解码器：RepeatVector + BiLSTM(32+32) + TimeDistributed(Dense)，重构原始时序
    
    Args:
        data (np.ndarray): 输入数据，形状为 (n_samples, timesteps, n_features)
                          - n_samples: 样本数量
                          - timesteps: 时间步长度（填充后的统一长度）
                          - n_features: 特征数量（单特征时为1）
        config (dict): 模型配置字典，包含以下键：
                      - latent_dim (int): 全局特征维度，默认64
                      - epochs (int): 训练轮数，默认50
                      - batch_size (int): 批量大小，默认32
                      - learning_rate (float): 学习率，默认0.001
                      - patience (int): 早停耐心值，默认5
    
    Returns:
        tuple: (features, training_history)
            features (np.ndarray): 提取的全局注意力特征，形状为 (n_samples, latent_dim)
            training_history (dict): 训练过程的历史信息，包含：
                - loss: 训练损失
                - val_loss: 验证损失
                - epochs_trained: 实际训练轮数
    
    Example:
        >>> import numpy as np
        >>> data = np.random.rand(100, 50, 1)  # 100个样本，50个时间步，1个特征
        >>> config = {"latent_dim": 64, "epochs": 50, "batch_size": 32, 
        ...           "learning_rate": 0.001, "patience": 5}
        >>> features, history = bilstm_ae_attention(data, config)
        >>> print(features.shape)  # (100, 64)
        >>> print(history.keys())  # dict_keys(['loss', 'val_loss', 'epochs_trained'])
    
    Note:
        全局注意力机制的优势：
        - 自动学习每个时间步的重要性，无需人工设计特征
        - 聚合时间信息，生成固定长度的全局特征，适合聚类、分类等任务
        - 减少特征维度，降低后续任务的计算复杂度
        - 保留时序数据的全局语义信息
        
        适用场景：
        - 聚类分析：需要固定长度特征的无监督学习任务
        - 分类任务：将时序数据映射到类别标签
        - 特征降维：将高维时序数据压缩为低维全局特征
    """
    # ===================== 1. 解析配置参数 =====================
    latent_dim = config["latent_dim"]  # 全局特征维度
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

    # ===================== 4. 构建 BiLSTM + 全局注意力自编码器模型 =====================
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

    # 全局注意力层：通过计算时间步权重并加权求和实现全局聚合
    # 步骤：
    # 1. 计算每个时间步的注意力权重（通过全连接层）
    # 2. 对权重进行softmax归一化
    # 3. 使用加权求和聚合时间维度
    
    # 计算注意力权重：使用全连接层计算每个时间步的重要性分数
    # 输出形状: (batch_size, timesteps, 1)
    attention_scores = Dense(1, activation='tanh', name="attention_scores")(encoder_bilstm)
    
    # 对注意力分数进行softmax归一化，确保所有时间步的权重和为1
    # 输出形状: (batch_size, timesteps, 1)
    attention_weights = Lambda(lambda x: K.softmax(x, axis=1), name="attention_weights")(attention_scores)
    
    # 使用加权求和聚合时间维度：将注意力权重与编码器输出相乘后求和
    # 输出形状: (batch_size, 64)，时间维度被聚合
    attention_output = Lambda(
        lambda x: K.sum(x[0] * x[1], axis=1),
        name="global_attention"
    )([encoder_bilstm, attention_weights])

    # 编码器特征降维：将 64 维全局特征降维到 latent_dim
    # - 输出形状: (batch_size, latent_dim)
    encoder_features = Dense(latent_dim, activation='relu', name="encoder_global_dense")(attention_output)

    # ===================== 解码器部分 =====================
    # 将全局特征复制到每个时间步，作为解码器的输入
    # - RepeatVector: 将 (batch_size, latent_dim) 转换为 (batch_size, timesteps, latent_dim)
    decoder_input = RepeatVector(timesteps, name="repeat_vector")(encoder_features)

    # BiLSTM 解码器：从全局特征重构时序数据
    # - 使用双向 LSTM（与编码器对称）
    # - return_sequences=True: 返回每个时间步的输出
    # - 输出形状: (batch_size, timesteps, 64)
    decoder_bilstm = Bidirectional(
        LSTM(32, activation='tanh', return_sequences=True),
        name="decoder_bilstm"
    )(decoder_input)

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
    lstm_autoencoder = Model(inputs=input_layer, outputs=output_layer, name="bilstm_global_attention_ae")
    
    # 编码器模型：用于特征提取，只保留编码器部分
    # - 输出形状: (batch_size, latent_dim)
    # - 这是全局特征，时间维度已被聚合
    lstm_encoder_model = Model(inputs=input_layer, outputs=encoder_features, name="global_attention_encoder")

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
    # 使用训练好的编码器提取全局特征
    # 输出形状: (n_samples, latent_dim)
    # 这是全局特征，时间维度已被聚合
    X_global_features = lstm_encoder_model.predict(X)

    # ===================== 10. 输出结果 =====================
    print(f"\n原始数据形状: {X.shape}")  # 输出 (n_samples, timesteps, n_features)
    print(f"全局注意力特征形状: {X_global_features.shape}")  # 输出 (n_samples, latent_dim)
    
    # 构建训练历史信息字典
    training_history = {
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'epochs_trained': len(history.history['loss']),
        'model_name': 'BiLSTM+Attention'
    }
    
    return X_global_features, training_history
