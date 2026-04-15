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


class GatingLayer(Layer):
    """
    门控层，参考DETSEC.py的gate函数实现
    
    使用sigmoid激活的全连接层生成门控掩码，
    用于控制信息流的通过程度。
    
    公式：gate(vec) = sigmoid(W @ vec + b)
    """
    def __init__(self, kernel_initializer='he_normal', **kwargs):  # 新增：统一初始化
        super(GatingLayer, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
    
    def build(self, input_shape):
        self.dense = Dense(
            input_shape[-1], 
            activation='sigmoid', 
            name='gate_dense',
            kernel_initializer=self.kernel_initializer  # 统一初始化
        )
        super(GatingLayer, self).build(input_shape)
    
    def call(self, inputs):
        return self.dense(inputs)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):  # 新增：配置序列化
        config = super(GatingLayer, self).get_config()
        config.update({
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
    BiLSTM + DETSEC风格全局注意力自编码器特征提取函数
    
    高优先级优化版：
    1. 输入维度校验 + 多归一化策略
    2. 编码器/解码器BiLSTM增加Dropout增强泛化
    3. 解码器新增门控层，与编码器结构对齐
    4. 补充学习率衰减回调
    5. 统一权重初始化 + 可选层归一化
    6. 鲁棒性优化（防止除零、梯度裁剪）
    
    模型结构：
    - 编码器：Input → Masking → BiLSTM(Dropout) → 门控 → 注意力 → 降维
    - 解码器：RepeatVector → BiLSTM(Dropout) → 门控 → TimeDistributed(Dense)
    
    Args:
        data (np.ndarray): 输入数据，形状为 (n_samples, timesteps, n_features)
        config (dict): 模型配置字典
    
    Returns:
        tuple: (features, training_history)
    """
    # ===================== 1. 输入维度校验（高优先级） =====================
    if len(data.shape) != 3:
        raise ValueError(f"输入数据需为3维 (样本数, 时间步, 特征数)，当前形状：{data.shape}")
    if data.shape[0] < 10:  # 最小样本数校验
        raise Warning(f"样本数过少（{data.shape[0]}），可能导致训练不稳定")
    
    # ===================== 2. 解析配置参数（扩展） =====================
    latent_dim = config.get("latent_dim", 64)
    epochs = config.get("epochs", 50)
    batch_size = config.get("batch_size", 32)
    learning_rate = config.get("learning_rate", 0.001)
    patience = config.get("patience", 5)
    attention_size = config.get("attention_size", 32)
    dropout_rate = config.get("dropout_rate", 0.2)  # 新增：Dropout率
    recurrent_dropout_rate = config.get("recurrent_dropout_rate", 0.1)  # 新增：循环Dropout率
    scaler_type = config.get("scaler_type", "minmax")  # 新增：归一化类型
    use_layer_norm = config.get("use_layer_norm", False)  # 新增：是否使用层归一化

    # ===================== 3. 提取数据维度信息 =====================
    n_samples, timesteps, n_features = data.shape
    
    # 赋值为模型输入
    X = data

    # ===================== 4. 数据归一化（增强鲁棒性） =====================
    X_scaled, scaler, scaled_mask_value = scale_data(X, scaler_type=scaler_type)
    print(f"归一化完成 | 类型: {scaler_type} | 范围: {X_scaled.min():.4f} ~ {X_scaled.max():.4f} | Mask值: {scaled_mask_value:.4f}")

    # ===================== 5. 构建 BiLSTM + DETSEC注意力自编码器模型 =====================
    # 输入层
    input_layer = Input(shape=(timesteps, n_features), name="input_layer")

    # Masking 层
    masking_layer = Masking(mask_value=scaled_mask_value, name="masking_layer")(input_layer)

    # ===================== 编码器部分（增加Dropout） =====================
    # BiLSTM 编码器（增加Dropout增强泛化）
    encoder_bilstm = Bidirectional(
        LSTM(
            32, 
            activation='tanh', 
            return_sequences=True,
            dropout=dropout_rate,  # 新增：输入Dropout
            recurrent_dropout=recurrent_dropout_rate,  # 新增：循环Dropout
            kernel_initializer=initializers.HeNormal()  # 统一初始化
        ),
        name="encoder_bilstm"
    )(masking_layer)
    # encoder_bilstm形状: (batch_size, timesteps, 64)
    
    # 可选：层归一化
    if use_layer_norm:
        encoder_bilstm = LayerNormalization(name="encoder_layer_norm")(encoder_bilstm)
    
    # 分离前向和后向输出
    forward_output = Lambda(lambda x: x[:, :, :32], name="forward_split")(encoder_bilstm)
    backward_output = Lambda(lambda x: x[:, :, 32:], name="backward_split")(encoder_bilstm)
    
    # 1. 对序列应用门控机制
    gate_fw_seq = GatingLayer(name="gate_forward_seq")(forward_output)
    gate_bw_seq = GatingLayer(name="gate_backward_seq")(backward_output)
    
    gated_fw_seq = Multiply(name="gated_forward_seq")([forward_output, gate_fw_seq])
    gated_bw_seq = Multiply(name="gated_backward_seq")([backward_output, gate_bw_seq])
    
    # 2. 对门控后的序列计算 Attention，聚合为向量
    attention_fw = Attention(
        attention_size=attention_size,
        kernel_initializer=initializers.HeNormal(),
        name="attention_forward"
    )(gated_fw_seq)
    
    attention_bw = Attention(
        attention_size=attention_size,
        kernel_initializer=initializers.HeNormal(),
        name="attention_backward"
    )(gated_bw_seq)
    
    # 3. 拼接融合后的特征
    encoder_concat = Concatenate(name="encoder_concat")([attention_fw, attention_bw])
    
    # 编码器特征降维（统一初始化）
    encoder_features = Dense(
        latent_dim, 
        activation='relu', 
        name="encoder_global_dense",
        kernel_initializer=initializers.HeNormal()
    )(encoder_concat)

    # ===================== 解码器部分（核心优化：门控对齐 + Dropout） =====================
    decoder_input = RepeatVector(timesteps, name="repeat_vector")(encoder_features)

    # 解码器BiLSTM（增加Dropout）
    decoder_bilstm = Bidirectional(
        LSTM(
            32, 
            activation='tanh', 
            return_sequences=True,
            dropout=dropout_rate,  # 新增：Dropout
            recurrent_dropout=recurrent_dropout_rate,  # 新增：循环Dropout
            kernel_initializer=initializers.HeNormal()  # 统一初始化
        ),
        name="decoder_bilstm"
    )(decoder_input)
    
    # 可选：层归一化
    if use_layer_norm:
        decoder_bilstm = LayerNormalization(name="decoder_layer_norm")(decoder_bilstm)
    
    # 核心优化：解码器新增门控层，与编码器结构对齐
    decoder_gate = GatingLayer(name="decoder_gate")(decoder_bilstm)
    decoder_gated = Multiply(name="decoder_gated_seq")([decoder_bilstm, decoder_gate])

    # 重构输出（保持linear激活，适配原始数据分布）
    output_layer = TimeDistributed(
        Dense(
            n_features, 
            activation='linear',
            kernel_initializer=initializers.HeNormal()  # 统一初始化
        ),
        name="output_layer"
    )(decoder_gated)

    # ===================== 6. 构建完整模型 =====================
    lstm_autoencoder = Model(inputs=input_layer, outputs=output_layer, name="bilstm_attention_ae_v5")
    lstm_encoder_model = Model(inputs=input_layer, outputs=encoder_features, name="attention_encoder_v5")
    
    # ===================== 7. 编译模型（增强鲁棒性） =====================
    optimizer = Adam(
        learning_rate=learning_rate, 
        clipnorm=1.0,  # 梯度裁剪，防止梯度爆炸
        epsilon=1e-7   # 数值稳定性
    )
    lstm_autoencoder.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]  # 新增：多指标监控
    )

    # ===================== 8. 配置回调（新增学习率衰减） =====================
    earliest_stop = EarlyStopping(
        monitor='val_loss',      
        patience=patience,       
        mode='min',              
        restore_best_weights=True,
        verbose=1                
    )
    
    # 新增：学习率衰减（高优先级）
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # 学习率衰减因子
        patience=3,  # 验证损失不下降则衰减
        min_lr=1e-6,  # 最小学习率
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
    print(f"\n训练完成 | 最终训练损失: {history.history['loss'][-1]:.4f} | 最终验证损失: {history.history['val_loss'][-1]:.4f}")
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
        'model_name': 'BiLSTM+Gated_Sequence+Attention (Optimized)'
    }
    
    return X_global_features, training_history


if __name__ == "__main__":
    # 测试代码
    print("测试优化版 BiLSTM+Gated+Attention 模型...")
    
    # 创建测试数据
    test_data = np.random.rand(100, 30, 5).astype(np.float32)  # 增加样本数和特征数，更贴近真实场景
    
    config = {
        "latent_dim": 16,
        "epochs": 10,
        "batch_size": 8,
        "learning_rate": 0.001,
        "patience": 3,
        "attention_size": 16,
        "dropout_rate": 0.2,          # 新增：Dropout配置
        "recurrent_dropout_rate": 0.1,# 新增：循环Dropout配置
        "scaler_type": "minmax",      # 新增：归一化类型
        "use_layer_norm": True        # 新增：层归一化
    }
    
    try:
        features, history = bilstm_ae_attention(test_data, config)
        print(f"\n提取的特征形状: {features.shape}")
        print(f"训练轮数: {history['epochs_trained']}")
        print(f"最终验证RMSE: {history['val_rmse'][-1]:.4f}")
        print("测试完成！")
    except Exception as e:
        print(f"测试出错: {e}")