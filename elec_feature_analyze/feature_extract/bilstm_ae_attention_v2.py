import os

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, RepeatVector, TimeDistributed, Dense, Masking,
    Bidirectional, Permute, Multiply, Lambda, Layer, Concatenate
)
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import initializers
import tensorflow as tf


class DETSECAttention(Layer):
    """
    参考DETSEC.py实现的注意力机制层
    
    实现细节：
    - 使用可学习的权重矩阵 W_omega 进行特征变换
    - 使用偏置项 b_omega 增加表达能力
    - 使用上下文向量 u_omega 计算注意力分数
    - 数学公式：
      1. v = tanh(inputs @ W_omega + b_omega)  # (batch, time, attention_size)
      2. vu = v @ u_omega                      # (batch, time)
      3. alphas = softmax(vu)                  # (batch, time)
      4. output = sum(inputs * alphas)         # (batch, features)
    
    Args:
        attention_size: 注意力隐藏层维度，默认32
        kernel_initializer: 权重初始化方法
    
    Returns:
        output: 加权聚合后的特征，形状为 (batch_size, nunits)
        alphas: 注意力权重，形状为 (batch_size, timesteps)
    """
    def __init__(self, attention_size=32, kernel_initializer='random_normal', **kwargs):
        super(DETSECAttention, self).__init__(**kwargs)
        self.attention_size = attention_size
        self.kernel_initializer = initializers.get(kernel_initializer)
        
    def build(self, input_shape):
        # input_shape: (batch_size, timesteps, nunits)
        self.nunits = input_shape[-1]
        self.timesteps = input_shape[1]
        
        # W_omega: (nunits, attention_size) - 特征变换矩阵
        self.W_omega = self.add_weight(
            name='W_omega',
            shape=(self.nunits, self.attention_size),
            initializer=self.kernel_initializer,
            trainable=True
        )
        
        # b_omega: (attention_size,) - 偏置项
        self.b_omega = self.add_weight(
            name='b_omega',
            shape=(self.attention_size,),
            initializer='zeros',
            trainable=True
        )
        
        # u_omega: (attention_size,) - 上下文向量
        self.u_omega = self.add_weight(
            name='u_omega',
            shape=(self.attention_size,),
            initializer=self.kernel_initializer,
            trainable=True
        )
        
        super(DETSECAttention, self).build(input_shape)
    
    def call(self, inputs):
        """
        前向传播
        
        Args:
            inputs: 编码器输出，形状为 (batch_size, timesteps, nunits)
        
        Returns:
            output: 注意力加权后的特征，形状为 (batch_size, nunits)
        """
        # inputs形状: (batch_size, timesteps, nunits)
        # batch_size = tf.shape(inputs)[0]
        
        # 第一步：计算 v = tanh(inputs @ W_omega + b_omega)
        # inputs @ W_omega: (batch, time, nunits) @ (nunits, attention_size) = (batch, time, attention_size)
        v = tf.tanh(tf.tensordot(inputs, self.W_omega, axes=1) + self.b_omega)
        # v形状: (batch_size, timesteps, attention_size)
        
        # 第二步：计算 vu = v @ u_omega
        # v @ u_omega: (batch, time, attention_size) @ (attention_size,) = (batch, time)
        vu = tf.tensordot(v, self.u_omega, axes=1)
        # vu形状: (batch_size, timesteps)
        
        # 第三步：计算 alphas = softmax(vu)
        alphas = tf.nn.softmax(vu, axis=1)  # 在时间维度上softmax
        # alphas形状: (batch_size, timesteps)
        
        # 第四步：加权求和 output = sum(inputs * alphas)
        # 扩展alphas维度: (batch, time) -> (batch, time, 1)
        alphas_expanded = tf.expand_dims(alphas, -1)
        # 加权: inputs * alphas_expanded = (batch, time, nunits) * (batch, time, 1) = (batch, time, nunits)
        weighted = inputs * alphas_expanded
        # 求和: (batch, time, nunits) -> (batch, nunits)
        output = tf.reduce_sum(weighted, axis=1)
        # output形状: (batch_size, nunits)
        
        # 保存alphas供后续可视化使用
        self.alphas = alphas
        
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])  # (batch_size, nunits)
    
    def get_config(self):
        config = super(DETSECAttention, self).get_config()
        config.update({
            'attention_size': self.attention_size,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        })
        return config


class GatingLayer(Layer):
    """
    门控层，参考DETSEC.py的gate函数实现
    
    使用sigmoid激活的全连接层生成门控掩码，
    用于控制信息流的通过程度。
    
    公式：gate(vec) = sigmoid(W @ vec + b)
    """
    def __init__(self, **kwargs):
        super(GatingLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.dense = Dense(input_shape[-1], activation='sigmoid', name='gate_dense')
        super(GatingLayer, self).build(input_shape)
    
    def call(self, inputs):
        return self.dense(inputs)
    
    def compute_output_shape(self, input_shape):
        return input_shape


def bilstm_ae_attention(data: np.ndarray, config: dict):
    """
    BiLSTM + DETSEC风格全局注意力自编码器特征提取函数
    
    修改版：先对BiLSTM输出序列进行门控（Gating），再进行注意力（Attention）聚合。
    
    模型结构：
    - 编码器：
      - BiLSTM(32+32)：提取双向时序特征
      - Split：分离前向和后向输出
      - GatingLayer：分别对前向和后向序列进行门控 (Sequence Level Gating)
      - DETSECAttention：分别对门控后的序列计算注意力，聚合为向量
      - Dense(latent_dim)：降维到目标维度
    - 解码器：RepeatVector + BiLSTM(32+32) + TimeDistributed(Dense)，重构原始时序
    
    Args:
        data (np.ndarray): 输入数据，形状为 (n_samples, timesteps, n_features)
        config (dict): 模型配置字典
    
    Returns:
        tuple: (features, training_history)
    """
    # ===================== 1. 解析配置参数 =====================
    latent_dim = config.get("latent_dim", 64)  # 全局特征维度
    epochs = config.get("epochs", 50)          # 训练轮数
    batch_size = config.get("batch_size", 32)  # 批量大小
    learning_rate = config.get("learning_rate", 0.001)  # 学习率
    patience = config.get("patience", 5)       # 早停耐心值
    attention_size = config.get("attention_size", 32)  # 注意力隐藏层维度

    # ===================== 2. 提取数据维度信息 =====================
    timesteps = data.shape[1]  # 时间步长度
    n_features = data.shape[2]  # 特征数量
    
    # 赋值为模型输入
    X = data

    # ===================== 3. 数据归一化 =====================
    X_min = X.min()
    X_max = X.max()
    X = (X - X_min) / (X_max - X_min + 1e-7)
    
    scaled_mask_value = (0.0 - X_min) / (X_max - X_min + 1e-7)
    print(f"归一化完成 | 范围: {X.min():.2f} ~ {X.max():.2f} | Mask值: {scaled_mask_value:.4f}")

    # ===================== 4. 构建 BiLSTM + DETSEC注意力自编码器模型 =====================
    # 输入层
    input_layer = Input(shape=(timesteps, n_features), name="input_layer")

    # Masking 层
    masking_layer = Masking(mask_value=scaled_mask_value, name="masking_layer")(input_layer)

    # ===================== 编码器部分 =====================
    # BiLSTM 编码器
    encoder_bilstm = Bidirectional(
        LSTM(32, activation='tanh', return_sequences=True),
        name="encoder_bilstm"
    )(masking_layer)
    # encoder_bilstm形状: (batch_size, timesteps, 64)
    
    # 分离前向和后向输出
    forward_output = Lambda(lambda x: x[:, :, :32], name="forward_split")(encoder_bilstm)
    backward_output = Lambda(lambda x: x[:, :, 32:], name="backward_split")(encoder_bilstm)
    
    # === 修改部分开始：先 Gating (Sequence Level) 再 Attention ===
    
    # 1. 对序列应用门控机制
    # 生成门控系数 (Batch, Time, 32)
    gate_fw_seq = GatingLayer(name="gate_forward_seq")(forward_output)
    gate_bw_seq = GatingLayer(name="gate_backward_seq")(backward_output)
    
    # 应用门控：Sequence * Gate
    gated_fw_seq = Multiply(name="gated_forward_seq")([forward_output, gate_fw_seq])
    gated_bw_seq = Multiply(name="gated_backward_seq")([backward_output, gate_bw_seq])
    
    # 2. 对门控后的序列计算 Attention，聚合为向量
    # 前向注意力 -> (Batch, 32)
    attention_fw = DETSECAttention(
        attention_size=attention_size,
        kernel_initializer=initializers.RandomNormal(stddev=0.1),
        name="attention_forward"
    )(gated_fw_seq)
    
    # 后向注意力 -> (Batch, 32)
    attention_bw = DETSECAttention(
        attention_size=attention_size,
        kernel_initializer=initializers.RandomNormal(stddev=0.1),
        name="attention_backward"
    )(gated_bw_seq)
    
    # 3. 拼接融合后的特征
    encoder_concat = Concatenate(name="encoder_concat")([attention_fw, attention_bw])
    # encoder_concat形状: (batch_size, 64)
    
    # === 修改部分结束 ===
    
    # 编码器特征降维
    encoder_features = Dense(latent_dim, activation='relu', name="encoder_global_dense")(encoder_concat)

    # ===================== 解码器部分 =====================
    decoder_input = RepeatVector(timesteps, name="repeat_vector")(encoder_features)

    decoder_bilstm = Bidirectional(
        LSTM(32, activation='tanh', return_sequences=True),
        name="decoder_bilstm"
    )(decoder_input)

    output_layer = TimeDistributed(
        Dense(n_features, activation='linear'),
        name="output_layer"
    )(decoder_bilstm)

    # ===================== 5. 构建完整模型 =====================
    lstm_autoencoder = Model(inputs=input_layer, outputs=output_layer, name="bilstm_detsec_attention_ae_v2")
    
    lstm_encoder_model = Model(inputs=input_layer, outputs=encoder_features, name="detsec_attention_encoder_v2")
    
    # ===================== 6. 编译模型 =====================
    lstm_autoencoder.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss='mse'
    )

    # ===================== 7. 配置早停回调 =====================
    earliest_stop = EarlyStopping(
        monitor='val_loss',      
        patience=patience,       
        mode='min',              
        restore_best_weights=True,
        verbose=1                
    )

    # ===================== 8. 训练模型 =====================
    history = lstm_autoencoder.fit(
        X, X,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.2,
        callbacks=[earliest_stop]
    )

    # ===================== 9. 提取特征 =====================
    X_global_features = lstm_encoder_model.predict(X)

    # ===================== 10. 输出结果 =====================
    print(f"\n原始数据形状: {X.shape}")
    print(f"DETSEC注意力特征形状: {X_global_features.shape}")
    
    training_history = {
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'epochs_trained': len(history.history['loss']),
        'model_name': 'BiLSTM+Gated_Sequence+Attention'
    }
    
    return X_global_features, training_history


if __name__ == "__main__":
    # 测试代码
    print("测试修改版 BiLSTM+Gated+Attention 模型...")
    
    # 创建测试数据
    test_data = np.random.rand(20, 30, 1).astype(np.float32)
    
    config = {
        "latent_dim": 16,
        "epochs": 5,
        "batch_size": 4,
        "learning_rate": 0.001,
        "patience": 3,
        "attention_size": 16
    }
    
    features, history = bilstm_ae_attention(test_data, config)
    print(f"\n提取的特征形状: {features.shape}")
    print(f"训练轮数: {history['epochs_trained']}")
    print("测试完成！")
