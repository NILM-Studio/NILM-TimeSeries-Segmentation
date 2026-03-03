# BiLSTM-Gated-Attention Autoencoder (V2) 模型文档

本文档详细说明了 `bilstm_ae_attention_v2.py` 中实现的改进版时间序列自编码器模型。该版本调整了特征处理的顺序，先对 LSTM 输出序列进行门控（Gating）处理，再通过 DETSEC Attention 机制聚合成全局特征。

## 1. 模型概述

该模型旨在从时间序列数据中提取鲁棒的全局特征。与 V1 版本相比，V2 版本采用了 **"先门控，后聚合"** 的策略。这意味着模型首先在时间步级别上筛选重要信息（门控），然后再计算加权和（注意力），从而生成更能代表序列关键信息的潜在向量。

**核心组件与流程：**
1.  **BiLSTM 编码器**：提取双向时序特征序列。
2.  **Sequence Gating (门控)**：在序列层面分别对前向和后向特征进行门控筛选。
3.  **DETSEC Attention (聚合)**：对门控后的序列进行注意力加权，聚合成固定长度的向量。
4.  **BiLSTM 解码器**：根据提取的全局特征重构原始时间序列。

---

## 2. 数学原理

### 2.1 总体流程

设 BiLSTM 输出的时间步序列为 $H = \{h_1, h_2, ..., h_T\}$。
我们将 $H$ 分离为前向流 $H_{fw}$ 和后向流 $H_{bw}$。

### 2.2 Sequence Gating (序列门控)

在 V2 版本中，门控机制直接作用于每个时间步的特征向量，用于抑制噪声或无关时间步的信息。

对于序列中的每个时刻 $t$ 的隐藏状态 $h_t$：

1.  **计算门控系数**：
    $$ g_t = \sigma(W_g h_t + b_g) $$
    其中：
    *   $\sigma$ 是 Sigmoid 激活函数，输出范围 $(0, 1)$。
    *   $W_g, b_g$ 是全连接层的权重和偏置。

2.  **应用门控**：
    $$ \tilde{h}_t = h_t \odot g_t $$
    其中 $\odot$ 表示逐元素乘法。这一步生成了“门控后的序列” $\tilde{H} = \{\tilde{h}_1, ..., \tilde{h}_T\}$。

### 2.3 DETSEC Attention (特征聚合)

注意力层接收门控后的序列 $\tilde{H}$，将其压缩为上下文向量 $s$。

1.  **特征变换**：
    $$ v_t = \tanh(W_\omega \tilde{h}_t + b_\omega) $$

2.  **计算注意力得分**：
    $$ score_t = v_t^T u_\omega $$

3.  **归一化权重**：
    $$ \alpha_t = \frac{\exp(score_t)}{\sum_{k=1}^{T} \exp(score_k)} $$

4.  **加权求和**：
    $$ s = \sum_{t=1}^{T} \alpha_t \tilde{h}_t $$
    注意：这里加权的对象是**门控后**的特征 $\tilde{h}_t$。

### 2.4 编码器特征融合全流程

在 `bilstm_ae_attention` (V2) 函数中，前向和后向特征的完整处理流程总结如下：

1.  **分离方向**：BiLSTM 输出 $H$ 分离为前向 $H_{fw}$ 和后向 $H_{bw}$。
2.  **独立序列门控 (Sequence Gating)**：
    $$ \tilde{H}_{fw} = \text{Gate}(H_{fw}) \odot H_{fw} $$
    $$ \tilde{H}_{bw} = \text{Gate}(H_{bw}) \odot H_{bw} $$
    *注：此处 Gate 输出的是与输入形状相同的序列级掩码。*
3.  **独立注意力聚合 (Attention Aggregation)**：
    $$ s_{fw} = \text{Attention}(\tilde{H}_{fw}) $$
    $$ s_{bw} = \text{Attention}(\tilde{H}_{bw}) $$
4.  **拼接与降维**：
    $$ z = \text{ReLU}(W_{dense}[s_{fw}; s_{bw}] + b_{dense}) $$
    其中 $z$ 即为最终提取的全局特征（Latent Vector）。

---

## 3. 代码结构解析

### 3.1 `DETSECAttention` 类

保持不变，负责将输入序列 $(Batch, Time, Features)$ 聚合为向量 $(Batch, Features)$。

### 3.2 `GatingLayer` 类

保持不变，负责生成门控系数。在 V2 中，它被应用于 3D 的序列输入 $(Batch, Time, Features)$，生成同样形状的门控掩码。

### 3.3 `bilstm_ae_attention` 函数 (V2 修改版)

**关键修改点 (Encoder 部分)：**

```python
# 1. 分离前向和后向输出
forward_output = Lambda(...)(encoder_bilstm)
backward_output = Lambda(...)(encoder_bilstm)

# 2. Sequence Level Gating (先门控)
# 生成针对每个时间步的门控系数
gate_fw_seq = GatingLayer(...)(forward_output)
gate_bw_seq = GatingLayer(...)(backward_output)

# 应用门控：Sequence * Gate
gated_fw_seq = Multiply()([forward_output, gate_fw_seq])
gated_bw_seq = Multiply()([backward_output, gate_bw_seq])

# 3. Attention Aggregation (后聚合)
# 输入是门控后的序列
attention_fw = DETSECAttention(...)(gated_fw_seq)
attention_bw = DETSECAttention(...)(gated_bw_seq)

# 4. 拼接
encoder_concat = Concatenate()([attention_fw, attention_bw])
```

**解码器部分**：
保持不变，使用 `RepeatVector` 将全局特征复制后通过 BiLSTM 进行重构。

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
from bilstm_ae_attention_v2 import bilstm_ae_attention

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

## 6. V1 与 V2 区别总结

| 特性 | V1 (bilstm_ae_attantion.py) | V2 (bilstm_ae_attention_v2.py) |
| :--- | :--- | :--- |
| **处理顺序** | BiLSTM $\rightarrow$ Attention $\rightarrow$ Gating | BiLSTM $\rightarrow$ Gating $\rightarrow$ Attention |
| **门控对象** | Attention 聚合后的**向量** ($s$) | LSTM 输出的**序列** ($h_t$) |
| **Attention 输入** | 原始 LSTM 输出 | 经过门控筛选后的 LSTM 输出 |
| **设计意图** | 对最终特征进行加权筛选 | 在聚合前抑制噪声时间步 |
