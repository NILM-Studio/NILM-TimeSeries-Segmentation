import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, RepeatVector, TimeDistributed, Dense, Masking,
    Bidirectional, Permute, Multiply, Lambda, Layer, Concatenate,
    LayerNormalization  # 新增：层归一化
)
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # 新增：学习率衰减
from tensorflow.keras import initializers
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler  # 新增：多归一化策略


class Attention(Layer):
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

    def __init__(self, attention_size=32, kernel_initializer='he_normal', **kwargs):  # 优化：默认HeNormal初始化
        super(Attention, self).__init__(**kwargs)
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
            initializer=self.kernel_initializer,  # 统一初始化
            trainable=True
        )

        super(Attention, self).build(input_shape)

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
        v = tf.tanh(tf.tensordot(inputs, self.W_omega, axes=1) + self.b_omega)
        # v形状: (batch_size, timesteps, attention_size)

        # 第二步：计算 vu = v @ u_omega
        vu = tf.tensordot(v, self.u_omega, axes=1)
        # vu形状: (batch_size, timesteps)

        # 第三步：计算 alphas = softmax(vu)
        alphas = tf.nn.softmax(vu, axis=1)  # 在时间维度上softmax
        # alphas形状: (batch_size, timesteps)

        # 第四步：加权求和 output = sum(inputs * alphas)
        alphas_expanded = tf.expand_dims(alphas, -1)
        weighted = inputs * alphas_expanded
        output = tf.reduce_sum(weighted, axis=1)
        # output形状: (batch_size, nunits)

        # 保存alphas供后续可视化使用
        self.alphas = alphas

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])  # (batch_size, nunits)

    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({
            'attention_size': self.attention_size,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        })
        return config


def scale_data(data: np.ndarray, scaler_type: str = "minmax", epsilon: float = 1e-7):
    """
    数据归一化函数（独立封装，增强复用性）

    Args:
        data: 输入数据 (n_samples, timesteps, n_features)
        scaler_type: 归一化类型，支持 minmax / standard / robust
        epsilon: 防止除零的小值

    Returns:
        scaled_data: 归一化后的数据
        scaler: 拟合后的归一化器
        scaled_mask_value: 原始0值对应的归一化后值
    """
    # 重塑为2D以适配sklearn scaler: (n_samples*timesteps, n_features)
    orig_shape = data.shape
    data_2d = data.reshape(-1, orig_shape[-1])

    # 选择归一化器
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"不支持的归一化类型: {scaler_type}，可选：minmax / standard / robust")

    # 拟合并转换
    scaled_data_2d = scaler.fit_transform(data_2d)
    scaled_data = scaled_data_2d.reshape(orig_shape)

    # 计算原始0值对应的归一化后值
    zero_2d = np.zeros((1, orig_shape[-1]))
    scaled_mask_value = scaler.transform(zero_2d)[0][0]  # 取第一个特征的0值映射

    return scaled_data, scaler, scaled_mask_value


def bilstm_ae_attention(data: np.ndarray, config: dict):
    """
    BiLSTM + DETSEC风格全局注意力自编码器特征提取函数（去除门控版）

    核心修改：
    1. 移除所有GatingLayer门控层，直接对BiLSTM输出计算注意力
    2. 解码器直接使用BiLSTM输出做重构，无门控加权
    3. 保留其他优化逻辑（Dropout、层归一化、学习率衰减等）

    模型结构：
    - 编码器：Input → Masking → BiLSTM(Dropout) → 注意力 → 降维
    - 解码器：RepeatVector → BiLSTM(Dropout) → TimeDistributed(Dense)

    Args:
        data (np.ndarray): 输入数据，形状为 (n_samples, timesteps, n_features)
        config (dict): 模型配置字典

    Returns:
        tuple: (features, training_history)
    """
    # ===================== 1. 输入维度校验 =====================
    if len(data.shape) != 3:
        raise ValueError(f"输入数据需为3维 (样本数, 时间步, 特征数)，当前形状：{data.shape}")
    if data.shape[0] < 10:  # 最小样本数校验
        raise Warning(f"样本数过少（{data.shape[0]}），可能导致训练不稳定")

    # ===================== 2. 解析配置参数 =====================
    latent_dim = config.get("latent_dim", 64)
    epochs = config.get("epochs", 50)
    batch_size = config.get("batch_size", 32)
    learning_rate = config.get("learning_rate", 0.001)
    patience = config.get("patience", 5)
    attention_size = config.get("attention_size", 32)
    dropout_rate = config.get("dropout_rate", 0.2)
    recurrent_dropout_rate = config.get("recurrent_dropout_rate", 0.1)
    scaler_type = config.get("scaler_type", "minmax")
    use_layer_norm = config.get("use_layer_norm", False)

    # ===================== 3. 提取数据维度信息 =====================
    n_samples, timesteps, n_features = data.shape
    X = data

    # ===================== 4. 数据归一化 =====================
    X_scaled, scaler, scaled_mask_value = scale_data(X, scaler_type=scaler_type)
    print(
        f"归一化完成 | 类型: {scaler_type} | 范围: {X_scaled.min():.4f} ~ {X_scaled.max():.4f} | Mask值: {scaled_mask_value:.4f}")

    # ===================== 5. 构建 BiLSTM + 注意力自编码器模型（无门控） =====================
    # 输入层
    input_layer = Input(shape=(timesteps, n_features), name="input_layer")

    # Masking 层
    masking_layer = Masking(mask_value=scaled_mask_value, name="masking_layer")(input_layer)

    # ===================== 编码器部分（移除门控） =====================
    # BiLSTM 编码器
    encoder_bilstm = Bidirectional(
        LSTM(
            32,
            activation='tanh',
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout_rate,
            kernel_initializer=initializers.HeNormal()
        ),
        name="encoder_bilstm"
    )(masking_layer)

    # 可选：层归一化
    if use_layer_norm:
        encoder_bilstm = LayerNormalization(name="encoder_layer_norm")(encoder_bilstm)

    # 分离前向和后向输出（无门控）
    forward_output = Lambda(lambda x: x[:, :, :32], name="forward_split")(encoder_bilstm)
    backward_output = Lambda(lambda x: x[:, :, 32:], name="backward_split")(encoder_bilstm)

    # 直接对BiLSTM输出计算Attention（核心修改：移除门控）
    attention_fw = Attention(
        attention_size=attention_size,
        kernel_initializer=initializers.HeNormal(),
        name="attention_forward"
    )(forward_output)

    attention_bw = Attention(
        attention_size=attention_size,
        kernel_initializer=initializers.HeNormal(),
        name="attention_backward"
    )(backward_output)

    # 拼接融合后的特征
    encoder_concat = Concatenate(name="encoder_concat")([attention_fw, attention_bw])

    # 编码器特征降维
    encoder_features = Dense(
        latent_dim,
        activation='relu',
        name="encoder_global_dense",
        kernel_initializer=initializers.HeNormal()
    )(encoder_concat)

    # ===================== 解码器部分（移除门控） =====================
    decoder_input = RepeatVector(timesteps, name="repeat_vector")(encoder_features)

    # 解码器BiLSTM
    decoder_bilstm = Bidirectional(
        LSTM(
            32,
            activation='tanh',
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout_rate,
            kernel_initializer=initializers.HeNormal()
        ),
        name="decoder_bilstm"
    )(decoder_input)

    # 可选：层归一化
    if use_layer_norm:
        decoder_bilstm = LayerNormalization(name="decoder_layer_norm")(decoder_bilstm)

    # 重构输出（核心修改：直接使用decoder_bilstm，无门控）
    output_layer = TimeDistributed(
        Dense(
            n_features,
            activation='linear',
            kernel_initializer=initializers.HeNormal()
        ),
        name="output_layer"
    )(decoder_bilstm)

    # ===================== 6. 构建完整模型 =====================
    lstm_autoencoder = Model(inputs=input_layer, outputs=output_layer, name="bilstm_attention_ae_v5_nogate")
    lstm_encoder_model = Model(inputs=input_layer, outputs=encoder_features, name="attention_encoder_v5_nogate")

    # ===================== 7. 编译模型 =====================
    optimizer = Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,  # 梯度裁剪
        epsilon=1e-7  # 数值稳定性
    )
    lstm_autoencoder.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )

    # ===================== 8. 配置回调 =====================
    earliest_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min',
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    callbacks = [earliest_stop, reduce_lr]

    # ===================== 9. 训练模型 =====================
    print("\n开始训练模型...")
    history = lstm_autoencoder.fit(
        X_scaled, X_scaled,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    # ===================== 10. 提取特征 =====================
    X_global_features = lstm_encoder_model.predict(X_scaled, verbose=0)

    # ===================== 11. 输出结果 =====================
    print(
        f"\n训练完成 | 最终训练损失: {history.history['loss'][-1]:.4f} | 最终验证损失: {history.history['val_loss'][-1]:.4f}")
    print(f"原始数据形状: {X.shape}")
    print(f"注意力特征形状: {X_global_features.shape}")

    training_history = {
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'mae': history.history.get('mae', []),
        'val_mae': history.history.get('val_mae', []),
        'rmse': history.history.get('rmse', []),
        'val_rmse': history.history.get('val_rmse', []),
        'epochs_trained': len(history.history['loss']),
        'model_name': 'BiLSTM+Attention (No Gating)'
    }

    return X_global_features, training_history


if __name__ == "__main__":
    # 测试代码
    print("测试去除门控的 BiLSTM+Attention 模型...")

    # 创建测试数据
    test_data = np.random.rand(100, 30, 5).astype(np.float32)

    config = {
        "latent_dim": 16,
        "epochs": 10,
        "batch_size": 8,
        "learning_rate": 0.001,
        "patience": 3,
        "attention_size": 16,
        "dropout_rate": 0.2,
        "recurrent_dropout_rate": 0.1,
        "scaler_type": "minmax",
        "use_layer_norm": True
    }

    try:
        features, history = bilstm_ae_attention(test_data, config)
        print(f"\n提取的特征形状: {features.shape}")
        print(f"训练轮数: {history['epochs_trained']}")
        print(f"最终验证RMSE: {history['val_rmse'][-1]:.4f}")
        print("测试完成！")
    except Exception as e:
        print(f"测试出错: {e}")