# BiLSTM+DETSEC注意力自编码器模型说明

## 1. 模型概述

本代码实现了一个基于双向长短期记忆网络（BiLSTM）和DETSEC风格注意力机制的自编码器模型，用于从时序数据中提取全局特征。该模型结合了双向LSTM的时序建模能力和注意力机制的特征加权能力，能够有效地从时序数据中学习到有代表性的全局特征。

### 1.1 模型应用场景

- 非侵入式负载监测（NILM）中的电器特征提取
- 时序数据异常检测
- 时序数据压缩与降维
- 传感器数据特征学习

### 1.2 模型核心优势

- **双向时序建模**：同时捕捉数据的前向和后向依赖关系
- **DETSEC注意力机制**：自适应地为不同时间步分配权重
- **门控融合策略**：智能融合双向注意力结果
- **自监督学习**：无需标注数据即可训练

## 2. 数学原理

### 2.1 DETSEC注意力机制

DETSEC注意力机制通过可学习的权重矩阵和上下文向量，计算每个时间步对最终特征的贡献度。

#### 2.1.1 数学公式

假设编码器输出为 $\mathbf{H} \in \mathbb{R}^{T \times D}$，其中 $T$ 是时间步长，$D$ 是隐藏层维度。

1. **特征变换**：将编码器输出映射到注意力空间
   $$\mathbf{V} = \tanh(\mathbf{H} \cdot \mathbf{W}_\omega + \mathbf{b}_\omega)$$
   - $\mathbf{W}_\omega \in \mathbb{R}^{D \times A}$：特征变换矩阵
   - $\mathbf{b}_\omega \in \mathbb{R}^{A}$：偏置项
   - $A$：注意力隐藏层维度

2. **注意力分数计算**：计算每个时间步的注意力权重
   $$\alpha_t = \text{softmax}(\mathbf{V} \cdot \mathbf{u}_\omega)$$
   - $\mathbf{u}_\omega \in \mathbb{R}^{A}$：上下文向量
   - $\alpha_t \in \mathbb{R}^{T}$：注意力权重，满足 $\sum_{t=1}^{T} \alpha_t = 1$

3. **加权聚合**：生成全局注意力特征
   $$\mathbf{h}_\alpha = \sum_{t=1}^{T} \alpha_t \cdot \mathbf{h}_t$$
   - $\mathbf{h}_\alpha \in \mathbb{R}^{D}$：注意力加权后的全局特征

#### 2.1.2 计算流程图

```
编码器输出 H (T×D)
    ↓
特征变换: tanh(H·W_ω + b_ω) → V (T×A)
    ↓
注意力分数: softmax(V·u_ω) → α (T)
    ↓
加权聚合: sum(α_t·h_t) → h_α (D)
```

### 2.2 门控融合机制

门控机制用于融合前向和后向注意力结果，通过sigmoid激活函数生成门控掩码，控制信息流的通过程度。

#### 2.2.1 数学公式

假设前向注意力结果为 $\mathbf{h}_{\alpha,f} \in \mathbb{R}^{D}$，后向注意力结果为 $\mathbf{h}_{\alpha,b} \in \mathbb{R}^{D}$：

1. **门控掩码计算**：
   $$\mathbf{g}_f = \sigma(\mathbf{W}_f \cdot \mathbf{h}_{\alpha,f} + \mathbf{b}_f)$$
   $$\mathbf{g}_b = \sigma(\mathbf{W}_b \cdot \mathbf{h}_{\alpha,b} + \mathbf{b}_b)$$
   - $\sigma$：sigmoid激活函数
   - $\mathbf{g}_f, \mathbf{g}_b \in \mathbb{R}^{D}$：门控掩码

2. **门控融合**：
   $$\mathbf{h}_{\text{fusion}} = \mathbf{g}_f \odot \mathbf{h}_{\alpha,f} + \mathbf{g}_b \odot \mathbf{h}_{\alpha,b}$$
   - $\odot$：逐元素乘法
   - $\mathbf{h}_{\text{fusion}} \in \mathbb{R}^{2D}$：融合后的特征

## 3. 模型架构

### 3.1 整体架构图

```
输入层 (T×F)
    ↓
Masking层 (处理不等长数据)
    ↓
编码器BiLSTM (输出: T×64)
    ↓
    ├─ 分离前向输出 (T×32) → DETSEC注意力 → 门控层 → 门控前向特征 (32)
    └─ 分离后向输出 (T×32) → DETSEC注意力 → 门控层 → 门控后向特征 (32)
        ↓
        拼接融合后的特征 (64)
            ↓
            编码器特征降维 → 全局特征 (latent_dim)
                ↓
                RepeatVector (T×latent_dim)
                    ↓
                    解码器BiLSTM (T×64)
                        ↓
                        TimeDistributed+Dense → 重构输出 (T×F)
```

### 3.2 各层详细说明

#### 3.2.1 输入层

- **输入形状**：$(n_{\text{samples}}, timesteps, n_{\text{features}})$
- **功能**：接收时序数据，支持多特征输入

#### 3.2.2 Masking层

- **功能**：处理不等长时序数据，忽略填充值
- **实现**：将填充值（默认0.0）替换为归一化后的掩码值

#### 3.2.3 BiLSTM编码器

- **配置**：双向LSTM，每个方向32个单元
- **输出形状**：$(batch, timesteps, 64)$
- **功能**：提取时序数据的双向依赖关系

#### 3.2.4 注意力机制

- **DETSECAttention层**：为每个时间步分配注意力权重
- **输出形状**：$(batch, 32)$（每个方向）

#### 3.2.5 门控融合层

- **GatingLayer层**：生成门控掩码
- **融合策略**：$\mathbf{h}_{\text{fusion}} = \mathbf{g}_f \odot \mathbf{h}_{\alpha,f} + \mathbf{g}_b \odot \mathbf{h}_{\alpha,b}$

#### 3.2.6 编码器特征降维

- **配置**：全连接层，激活函数为ReLU
- **输出形状**：$(batch, latent_dim)$
- **功能**：将融合后的特征降维到目标维度

#### 3.2.7 解码器

- **RepeatVector**：将全局特征复制到每个时间步
- **BiLSTM解码器**：从全局特征重构时序数据
- **TimeDistributed+Dense**：输出层，重构原始特征

## 4. 代码结构

### 4.1 自定义层实现

#### 4.1.1 DETSECAttention类

```python
class DETSECAttention(Layer):
    def __init__(self, attention_size=32, kernel_initializer='random_normal', **kwargs):
        # 初始化注意力层参数
        # ...
    
    def build(self, input_shape):
        # 构建可学习权重
        # W_omega: (nunits, attention_size)
        # b_omega: (attention_size,)
        # u_omega: (attention_size,)
        # ...
    
    def call(self, inputs):
        # 前向传播计算
        # 1. 特征变换: v = tanh(inputs @ W_omega + b_omega)
        # 2. 注意力分数: vu = v @ u_omega
        # 3. 权重归一化: alphas = softmax(vu)
        # 4. 加权聚合: output = sum(inputs * alphas)
        # ...
```

#### 4.1.2 GatingLayer类

```python
class GatingLayer(Layer):
    def build(self, input_shape):
        # 构建门控层权重
        self.dense = Dense(input_shape[-1], activation='sigmoid', name='gate_dense')
        # ...
    
    def call(self, inputs):
        # 生成门控掩码: gate(vec) = sigmoid(W @ vec + b)
        return self.dense(inputs)
```

### 4.2 主函数bilstm_ae_attention

```python
def bilstm_ae_attention(data: np.ndarray, config: dict):
    # 1. 解析配置参数
    # 2. 提取数据维度信息
    # 3. 数据归一化
    # 4. 构建BiLSTM+DETSEC注意力自编码器模型
    # 5. 编译模型
    # 6. 配置早停回调
    # 7. 训练模型
    # 8. 提取特征
    # 9. 输出结果
    # ...
```

## 5. 代码执行流程

### 5.1 数据预处理

1. **数据归一化**：使用Min-Max归一化将数据映射到[0, 1]区间
   $$X_{\text{norm}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}} + \epsilon}$$

2. **掩码值计算**：计算归一化后的填充值，用于Masking层
   $$\text{scaled_mask_value} = \frac{0.0 - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}} + \epsilon}$$

### 5.2 模型训练

1. **编译模型**：使用Adam优化器，MSE损失函数，梯度裁剪防止梯度爆炸
2. **早停策略**：监控验证集损失，当patience个epoch无改善时停止训练
3. **自监督学习**：输入和标签都是原始数据，模型学习重构输入

### 5.3 特征提取

1. **编码器模型**：使用训练好的编码器部分提取全局特征
2. **输出形状**：$(n_{\text{samples}}, latent_dim)$
3. **特征性质**：融合了双向时序信息和注意力权重

## 6. 输入输出示例

### 6.1 输入数据格式

```python
# 输入数据形状: (n_samples, timesteps, n_features)
# 示例：20个样本，每个样本30个时间步，1个特征
test_data = np.random.rand(20, 30, 1).astype(np.float32)
```

### 6.2 配置参数

```python
config = {
    "latent_dim": 16,        # 全局特征维度
    "epochs": 5,             # 训练轮数
    "batch_size": 4,         # 批量大小
    "learning_rate": 0.001,  # 学习率
    "patience": 3,           # 早停耐心值
    "attention_size": 16     # 注意力隐藏层维度
}
```

### 6.3 输出结果

```python
features, history = bilstm_ae_attention(test_data, config)

# 输出示例
原始数据形状: (20, 30, 1)
DETSEC注意力特征形状: (20, 16)
提取的特征形状: (20, 16)
训练轮数: 5
```

## 7. 模型训练历史

训练过程中记录的指标包括：

- **loss**：训练集损失
- **val_loss**：验证集损失
- **epochs_trained**：实际训练轮数
- **model_name**：模型名称

## 8. 代码优化建议

### 8.1 超参数调优

- **attention_size**：建议在8-64之间调整
- **latent_dim**：建议在16-128之间调整
- **BiLSTM单元数**：可根据数据复杂度调整

### 8.2 模型扩展

- 添加更多注意力头，实现多头注意力
- 尝试不同的注意力机制变体
- 结合卷积层提取局部特征

### 8.3 训练优化

- 使用学习率衰减策略
- 尝试不同的优化器（如RMSprop、AdamW）
- 增加数据增强，提高模型泛化能力

## 9. 代码使用说明

### 9.1 基本使用

```python
import numpy as np
from bilstm_ae_attention import bilstm_ae_attention

# 准备数据
data = np.random.rand(100, 50, 1)  # 100个样本，50个时间步，1个特征

# 配置参数
config = {
    "latent_dim": 64,
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001,
    "patience": 5,
    "attention_size": 32
}

# 训练模型并提取特征
features, history = bilstm_ae_attention(data, config)
```

### 9.2 与其他模型结合

```python
# 提取特征后可用于分类、聚类等任务
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(features)
```

## 10. 总结

本代码实现的BiLSTM+DETSEC注意力自编码器模型是一种强大的时序特征提取工具，结合了双向LSTM的时序建模能力和注意力机制的特征加权能力。该模型能够自适应地学习时序数据中的重要特征，为后续的数据分析和任务提供有力支持。

通过合理调整超参数和模型结构，该模型可以应用于多种时序数据处理场景，尤其是在非侵入式负载监测等领域具有广阔的应用前景。