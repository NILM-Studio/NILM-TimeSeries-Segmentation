# BiLSTM-Attention-Sum Autoencoder (V3) 模型文档

本文档详细说明了 `bilstm_ae_attention_v3.py` 中实现的第三版时间序列自编码器模型。该版本对特征融合机制进行了简化和调整，采用了前向和后向注意力向量的加权和作为最终的全局特征，移除了额外的全连接映射层。

## 1. 模型概述

V3 版本旨在探索一种更直接的特征融合方式。与 V1（拼接+Dense）和 V2（先门控+后聚合）不同，V3 回归到 V1 的“先聚合，后门控”流程，但在最终融合阶段放弃了拼接（Concatenation），转而使用**加法（Addition）**。这种设计假设前向和后向的语义空间是对齐的，可以直接叠加增强。

**核心组件与流程：**
1.  **BiLSTM 编码器**：提取双向时序特征。
2.  **DETSEC Attention**：分别对前向和后向 LSTM 输出计算注意力向量。
3.  **Gating Mechanism**：分别对前向和后向注意力向量应用门控。
4.  **Sum Fusion (加法融合)**：将门控后的前向和后向向量直接相加，得到全局特征。
5.  **BiLSTM 解码器**：根据融合后的特征重构原始序列。

---

## 2. 数学原理

### 2.1 总体流程

设 BiLSTM 输出的时间步序列为 $H = \{h_1, h_2, ..., h_T\}$。
我们将 $H$ 分离为前向流 $H_{fw}$ 和后向流 $H_{bw}$。

### 2.2 独立注意力聚合

首先，分别计算前向和后向的上下文向量：

$$ s_{fw} = \text{Attention}(H_{fw}) $$
$$ s_{bw} = \text{Attention}(H_{bw}) $$

其中 $\text{Attention}(\cdot)$ 的计算过程与 V1/V2 保持一致（$v_t = \tanh(...)$, $\alpha_t = \text{softmax}(...)$）。

### 2.3 门控与加法融合 (Gated Sum Fusion)

这是 V3 版本最核心的改动。

1.  **计算门控系数**：
    $$ g_{fw} = \sigma(W_{g} s_{fw} + b_{g}) $$
    $$ g_{bw} = \sigma(W_{g} s_{bw} + b_{g}) $$
    *注：这里前向和后向使用独立的参数（代码中实例化了两个 GatingLayer）。*

2.  **应用门控并融合**：
    最终的全局特征 $z$ 由下式计算：

    $$ z = (g_{fw} \odot s_{fw}) + (g_{bw} \odot s_{bw}) $$

    *   $\odot$ 表示逐元素乘法。
    *   $+$ 表示逐元素加法。
    *   **移除**了 V1 中的 $\text{ReLU}(W_{dense}[\cdot] + b_{dense})$ 操作。

这种融合方式要求 $s_{fw}$ 和 $s_{bw}$ 的维度必须一致（由 LSTM 单元数决定，代码中固定为 64）。

---

## 3. 代码结构解析

### 3.1 `bilstm_ae_attention` 函数 (V3 修改版)

**关键修改点 (Encoder 部分)：**

```python
# 1. 强制 LSTM 单元数
lstm_units = 64  # 为了保证相加后的维度为 64，这里需设为 64 (Concatenation 版本中是 32+32=64)

# 2. Attention 计算 (同 V1)
attention_fw = DETSECAttention(...)(forward_output)
attention_bw = DETSECAttention(...)(backward_output)

# 3. Gating 计算 (同 V1)
gate_fw = GatingLayer(...)(attention_fw)
gate_bw = GatingLayer(...)(attention_bw)

# 4. 门控加权
gated_fw = Multiply()([gate_fw, attention_fw])
gated_bw = Multiply()([gate_bw, attention_bw])

# 5. 加法融合 (Sum Fusion) - V3 特有
# 移除了 Concatenate 和 Dense 层
encoder_features = Add()([gated_fw, gated_bw])
# 输出形状: (Batch, 64)
```

**维度说明**：
为了使 V3 版本的最终输出维度（即 `latent_dim`）与 V1/V2 版本保持一致（均为 64），V3 版本中 BiLSTM 的单向单元数被增加到了 **64**（V1/V2 中为 32）。
这是因为：
*   **V1/V2**: `Concat([32, 32])` $\rightarrow$ 64
*   **V3**: `Add([64, 64])` $\rightarrow$ 64

## 4. 参数配置 (`config` 字典)

| 参数名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `latent_dim` | **失效** | 在 V3 中由 LSTM 单元数决定 (固定为 64) |
| `epochs` | 50 | 最大训练轮数 |
| `batch_size` | 32 | 批处理大小 |
| `learning_rate` | 0.001 | 优化器学习率 |
| `patience` | 5 | 早停机制的耐心值 |
| `attention_size` | 32 | 注意力机制内部变换的维度 |

## 5. V1, V2, V3 版本对比

| 特性 | V1 (Concat + Dense) | V2 (Sequence Gating) | V3 (Sum Fusion) |
| :--- | :--- | :--- | :--- |
| **处理顺序** | LSTM $\rightarrow$ Att $\rightarrow$ Gate | LSTM $\rightarrow$ Gate $\rightarrow$ Att | LSTM $\rightarrow$ Att $\rightarrow$ Gate |
| **融合方式** | Concatenate + Dense (ReLU) | Concatenate + Dense (ReLU) | **Add (加法)** |
| **Latent Dim** | 可通过 Dense 层任意指定 | 可通过 Dense 层任意指定 | **固定** (等于 LSTM 单元数) |
| **参数量** | 较多 (含融合层参数) | 较多 (含融合层参数) | **最少** (无融合层参数) |
| **非线性** | 强 (ReLU) | 强 (ReLU) | 弱 (仅依赖 Gate) |
