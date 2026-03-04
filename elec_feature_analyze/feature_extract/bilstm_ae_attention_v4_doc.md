# BiLSTM-Shared-Attention Autoencoder (V4) 模型文档

本文档详细说明了 `bilstm_ae_attention_v4.py` 中实现的第四版时间序列自编码器模型。该版本在 V3 的基础上，进一步引入了**参数共享 (Parameter Sharing)** 机制。通过让前向和后向分支复用同一套 Attention 和 Gating 参数，模型在保持双向建模能力的同时，显著减少了参数量，并强化了特征的统一表示能力。

## 1. 核心设计理念

**“双向性” 的本质是无方向的全局关联**

正如您指出的，在 Transformer 的双向自注意力（Bidirectional Self-Attention）或 BiDAF 等模型中，并不存在“前向参数”和“反向参数”之分。
*   $Q, K, V$ 矩阵由同一套 $W_Q, W_K, W_V$ 生成。
*   相似度计算 $S = QK^T$ 覆盖了 $j < t$（历史/前向）和 $j > t$（未来/后向）的所有上下文。

基于此理念，V4 版本假设：
**“判断某个时间步是否重要的标准（Attention），以及判断某个特征向量是否需要保留的标准（Gating），在时间轴的正向和反向上应该是通用的。”**

因此，我们移除了 V3 中独立的 `attention_fw/bw` 和 `gate_fw/bw`，改为：
1.  **Shared Attention**: 一套 $W_\omega, b_\omega, u_\omega$ 同时服务于前向序列 $H_{fw}$ 和后向序列 $H_{bw}$。
2.  **Shared Gating**: 一套 $W_{gate}, b_{gate}$ 同时服务于前向向量 $s_{fw}$ 和后向向量 $s_{bw}$。

## 2. 数学原理

### 2.1 共享注意力聚合

设 BiLSTM 输出的前向序列为 $H_{fw}$，后向序列为 $H_{bw}$。
定义共享的注意力函数 $\text{Attn}_{\theta}(\cdot)$，参数为 $\theta = \{W_\omega, b_\omega, u_\omega\}$。

$$ s_{fw} = \text{Attn}_{\theta}(H_{fw}) $$
$$ s_{bw} = \text{Attn}_{\theta}(H_{bw}) $$

注意：这里 $s_{fw}$ 和 $s_{bw}$ 是使用**同一组参数**计算出来的。

### 2.2 共享门控与加法融合

定义共享的门控函数 $\text{Gate}_{\phi}(\cdot)$，参数为 $\phi = \{W_{gate}, b_{gate}\}$。

1.  **计算门控系数**：
    $$ g_{fw} = \text{Gate}_{\phi}(s_{fw}) = \sigma(W_{gate} s_{fw} + b_{gate}) $$
    $$ g_{bw} = \text{Gate}_{\phi}(s_{bw}) = \sigma(W_{gate} s_{bw} + b_{gate}) $$

2.  **融合 (Sum Fusion)**：
    $$ z = (g_{fw} \odot s_{fw}) + (g_{bw} \odot s_{bw}) $$

---

## 3. 代码结构解析

### 3.1 `bilstm_ae_attention` 函数 (V4 修改版)

**关键修改点 (Encoder 部分)：**

```python
# 1. 实例化共享层 (只创建一次)
shared_attention_layer = DETSECAttention(..., name="shared_attention")
shared_gating_layer = GatingLayer(name="shared_gate")

# 2. 复用共享层 (调用两次)
# 前向分支
attention_fw = shared_attention_layer(forward_output)
gate_fw = shared_gating_layer(attention_fw)

# 后向分支 (使用完全相同的层对象)
attention_bw = shared_attention_layer(backward_output)
gate_bw = shared_gating_layer(attention_bw)

# 3. 加法融合 (同 V3)
encoder_features = Add()([gated_fw, gated_bw])
```

### 3.2 参数量对比

假设 `latent_dim` (LSTM 单元数) = 64, `attention_size` = 32。

*   **Attention 层参数**: $W_\omega (64 \times 32) + b_\omega (32) + u_\omega (32) = 2112$
    *   V3 (独立): $2112 \times 2 = 4224$
    *   V4 (共享): $2112 \times 1 = \mathbf{2112}$ (减少 50%)

*   **Gating 层参数**: $W_{gate} (64 \times 64) + b_{gate} (64) = 4160$
    *   V3 (独立): $4160 \times 2 = 8320$
    *   V4 (共享): $4160 \times 1 = \mathbf{4160}$ (减少 50%)

**总结**：在注意力机制和门控机制部分，V4 版本的参数量只有 V3 版本的一半，这有助于减轻过拟合风险，尤其是在数据量较小的情况下。

## 4. 版本演进总结

| 版本 | 融合结构 | 参数策略 | Latent Dim | 特点 |
| :--- | :--- | :--- | :--- | :--- |
| **V1** | Concat + Dense | 独立 | 可变 | 经典结构，参数最多，非线性强 |
| **V2** | Seq Gate + Attn | 独立 | 可变 | 引入序列级门控，先筛选后聚合 |
| **V3** | **Sum Fusion** | 独立 | 固定 | 移除 Dense 层，直接加法融合 |
| **V4** | **Sum Fusion** | **共享** | 固定 | **参数共享**，符合双向注意力本质 |
