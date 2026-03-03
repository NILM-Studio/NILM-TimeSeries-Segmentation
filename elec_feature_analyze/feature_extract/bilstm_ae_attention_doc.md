# BiLSTM-DETSEC Attention Autoencoder 模型文档

本文档详细说明了 `bilstm_ae_attantion.py` 中实现的基于 BiLSTM 和 DETSEC 风格注意力机制的时间序列自编码器模型。

## 1. 模型概述

该模型是一个用于时间序列特征提取的自编码器（Autoencoder）。它结合了双向长短期记忆网络（BiLSTM）来捕获时间序列的时序依赖性，并引入了 DETSEC（Deep Energy Time-Series Energy Classification）风格的注意力机制和门控机制，以增强模型对关键时间步的关注能力和特征融合能力。

**核心组件：**
1.  **BiLSTM 编码器**：提取双向时序特征。
2.  **DETSEC Attention**：分别对前向和后向 LSTM 的输出计算注意力加权特征。
3.  **Gating Layer**：使用门控机制对注意力特征进行加权。
4.  **BiLSTM 解码器**：根据提取的全局特征重构原始时间序列。

---

## 2. 数学原理

### 2.1 DETSEC Attention 机制

注意力层的目标是将 LSTM 输出的时间步序列 $H = \{h_1, h_2, ..., h_T\}$ 聚合成一个固定长度的上下文向量 $s$。

代码实现类：`DETSECAttention`

**公式推导：**

假设输入 $h_t$ 是时刻 $t$ 的隐藏状态（代码中的 `inputs`）。

1.  **特征变换**：首先通过一个单层神经网络将隐藏状态变换到注意力空间。
    $$ v_t = \tanh(W_\omega h_t + b_\omega) $$
    其中：
    *   $W_\omega$ 是权重矩阵 (形状: `nunits` $\times$ `attention_size`)
    *   $b_\omega$ 是偏置向量 (形状: `attention_size`)
    *   $v_t$ 是变换后的特征向量

2.  **计算注意力得分**：将变换后的特征与一个可学习的上下文向量 $u_\omega$ 进行点积，衡量重要性。
    $$ score_t = v_t^T u_\omega $$
    其中：
    *   $u_\omega$ 是上下文向量 (形状: `attention_size`)

3.  **归一化权重**：使用 Softmax 函数将得分归一化为概率分布 $\alpha_t$。
    $$ \alpha_t = \frac{\exp(score_t)}{\sum_{k=1}^{T} \exp(score_k)} $$

4.  **加权求和**：根据权重对所有时间步的隐藏状态进行加权求和，得到最终的上下文向量 $s$。
    $$ s = \sum_{t=1}^{T} \alpha_t h_t $$

### 2.2 Gating 机制 (门控融合)

为了控制信息流，模型对前向和后向的注意力向量分别应用了门控机制。

代码实现类：`GatingLayer`

**公式：**

对于输入向量 $s$（即上面的注意力输出），门控值 $g$ 计算如下：

$$ g = \sigma(W_g s + b_g) $$

其中：
*   $\sigma$ 是 Sigmoid 激活函数，输出范围 $(0, 1)$。
*   $W_g, b_g$ 是全连接层的权重和偏置。

最终的门控输出为逐元素相乘：
$$ s_{gated} = g \odot s $$

### 2.3 编码器特征融合

在 `bilstm_ae_attention` 函数中，前向和后向特征的处理流程如下：

1.  **分离方向**：BiLSTM 输出 $H$ 分离为前向 $H_{fw}$ 和后向 $H_{bw}$。
2.  **独立注意力**：
    $$ s_{fw} = \text{Attention}(H_{fw}) $$
    $$ s_{bw} = \text{Attention}(H_{bw}) $$
3.  **独立门控**：
    $$ \hat{s}_{fw} = \text{Gate}(s_{fw}) \odot s_{fw} $$
    $$ \hat{s}_{bw} = \text{Gate}(s_{bw}) \odot s_{bw} $$
4.  **拼接与降维**：
    $$ z = \text{ReLU}(W_{dense}[\hat{s}_{fw}; \hat{s}_{bw}] + b_{dense}) $$
    其中 $z$ 即为最终提取的全局特征（Latent Vector）。

---

## 3. 代码结构解析

### 3.1 `DETSECAttention` 类

继承自 `keras.layers.Layer`，实现了上述的注意力机制。

*   **`build`**: 初始化可学习参数 `W_omega`, `b_omega`, `u_omega`。
*   **`call`**: 执行前向传播计算。
    *   利用 `tf.tensordot` 进行矩阵乘法。
    *   利用 `tf.nn.softmax` 计算权重。
    *   保存 `self.alphas` 以便后续可视化分析。

### 3.2 `GatingLayer` 类

简单的门控层。

*   **`build`**: 定义一个 Dense 层，激活函数为 Sigmoid。
*   **`call`**: 直接调用 Dense 层输出门控系数。
    *   *注意：代码中实际在 Lambda 层完成了乘法操作，此类仅生成门控系数 $g$。*

### 3.3 `bilstm_ae_attention` 函数

构建和训练整个模型的全流程函数。

**主要步骤：**

1.  **数据预处理**：
    *   执行 Min-Max 归一化，将数据缩放到 $[0, 1]$。
    *   计算 Masking 值，处理填充数据。

2.  **构建编码器 (Encoder)**：
    *   `Masking`: 屏蔽填充值。
    *   `Bidirectional(LSTM)`: 提取时序特征。
    *   `Lambda` (Split): 将双向输出拆分为前向和后向流。
    *   `DETSECAttention`: 双路并行计算注意力。
    *   `GatingLayer` + `Lambda` (Multiply): 双路并行应用门控。
    *   `Concatenate`: 拼接双向结果。
    *   `Dense`: 映射到最终的 `latent_dim`。

3.  **构建解码器 (Decoder)**：
    *   `RepeatVector`: 将 2D 特征向量复制扩展为 3D 时序输入。
    *   `Bidirectional(LSTM)`: 对应编码器的结构，尝试恢复序列。
    *   `TimeDistributed(Dense)`: 对每个时间步独立重构特征。

4.  **模型编译与训练**：
    *   Loss: MSE (均方误差)。
    *   Optimizer: Adam (带梯度裁剪 `clipnorm=1.0`)。
    *   Callback: EarlyStopping (防止过拟合)。

5.  **特征提取**：
    *   构建子模型 `lstm_encoder_model`，输入与主模型相同，输出为编码器的 `encoder_features` 层。
    *   调用 `predict` 获取全局特征。

## 4. 参数配置 (`config` 字典)

| 参数名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `latent_dim` | 64 | 最终提取的全局特征向量的维度 |
| `epochs` | 50 | 最大训练轮数 |
| `batch_size` | 32 | 批处理大小 |
| `learning_rate` | 0.001 | 优化器学习率 |
| `patience` | 5 | 早停机制的耐心值 |
| `attention_size` | 32 | 注意力机制内部变换的维度 |

## 5. 使用示例

```python
import numpy as np
from bilstm_ae_attantion import bilstm_ae_attention

# 1. 准备数据: (样本数, 时间步, 特征数)
data = np.random.rand(100, 50, 1)

# 2. 配置参数
config = {
    "latent_dim": 32,
    "epochs": 20,
    "batch_size": 16,
    "attention_size": 16
}

# 3. 运行模型并提取特征
features, history = bilstm_ae_attention(data, config)

print("提取的特征形状:", features.shape) 
# 输出: (100, 32)
```
