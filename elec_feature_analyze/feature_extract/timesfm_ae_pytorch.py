import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from timesfm.timesfm_torch import TimesFmTorch as TimesFm

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

print(f"✅ PyTorch版本: {torch.__version__}")
print(f"✅ CUDA可用: {torch.cuda.is_available()}")
device_flag = "gpu" if torch.cuda.is_available() else "cpu"

# ===================================== 2. Hyperparameter=====================================
latent_dim = 8  # 最终特征维度
epochs = 50  # 训练轮数
batch_size = 32  # batch大小
learning_rate = 0.001  # 学习率
selected_feature_idx = 0  # 单特征输入
n_clusters = 3  # 聚类数量

# ===================================== 3. Load Data and Preprocess】=====================================
data = np.load("../time_clustering/cluster_data/data.npy")
seq_len = np.load("../time_clustering/cluster_data/seq_length.npy")
# 单特征切片：shape=(n_samples, timesteps, 1)
data_single = data[:, :, selected_feature_idx:selected_feature_idx + 1]
n_samples, timesteps, n_features = data_single.shape
print(f"\n数据加载完成: 样本数={n_samples}, 时间步={timesteps}, 特征数={n_features}")

# 数据预处理：转Tensor + 不等长时序MASK(填充0值失效)
data_tensor = torch.from_numpy(data_single).float()
mask = torch.zeros_like(data_tensor)
for i in range(n_samples):
    mask[i, :seq_len[i], :] = 1.0
data_tensor = data_tensor * mask

# 切分训练/验证集 (20%验证集，和原逻辑一致)
val_size = int(0.2 * n_samples)
train_x = data_tensor[:-val_size]
val_x = data_tensor[-val_size:]

# ===================================== 4. ✅ 官方TimesFm 正确初始化【严格对齐你贴的源码，无任何错误】=====================================
timesfm = TimesFm(
    context_len=512,  # 上下文长度
    horizon_len=timesteps,  # 预测视野长度，等于你的时间步
    input_patch_len=timesteps,  # 输入分块长度=你的时间步
    output_patch_len=timesteps,  # 输出分块长度=输入长度
    num_layers=6,  # 层数
    num_heads=4,  # 头数，满足 32%4=0
    model_dims=32,  # 模型核心维度
    quantiles=[0.1, 0.5, 0.9],  # 分位数
    per_core_batch_size=batch_size,  # 单卡batch大小
    backend=device_flag,  # cpu/gpu自动适配
    use_pos_emb=True  # 开启位置编码
)
print("\nTimesFmTorch初始化成功！")


# ===================================== 5. TimesFm+自编码器封装【适配无监督训练+特征提取】=====================================
class TimesFMAutoEncoder(nn.Module):
    def __init__(self, timesfm_model, timesteps, n_features, latent_dim):
        super().__init__()
        self.timesfm = timesfm_model
        self.timesteps = timesteps
        self.n_features = n_features
        self.latent_dim = latent_dim

        # 特征提取投影层：时序输出 → 样本级低维特征
        self.encoder_proj = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

        # 重构层：低维特征 → 还原时序数据
        self.decoder_proj = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_features)
        )

        # 核心：冻结官方TimesFm预训练权重，只训练投影层，不破坏预训练效果
        for param in self.timesfm.parameters():
            param.requires_grad = False

    def encode(self, x):
        """特征提取：官方forecast输出 + 全局平均池化 → (batch_size, latent_dim)"""
        with torch.no_grad():  # 官方模型推理无梯度
            forecast_out = self.timesfm.forecast(x)  # 官方唯一调用方法，输出和输入同形状
        feat = torch.mean(forecast_out, dim=1)  # 时序维度池化 → 样本级特征
        feat = self.encoder_proj(feat)
        return feat

    def forward(self, x):
        """无监督重构：特征提取 → 时序还原"""
        feat = self.encode(x)
        # 特征扩维：(batch, latent_dim) → (batch, timesteps, latent_dim)
        feat_expand = feat.unsqueeze(1).repeat(1, self.timesteps, 1)
        recon_x = self.decoder_proj(feat_expand)
        return recon_x


# 初始化模型
model = TimesFMAutoEncoder(timesfm, timesteps, n_features, latent_dim)
print("✅ TimesFM自编码器封装完成！")

# ===================================== 6. Train Config【MSE Loss+Adam Optimization】==================================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ===================================== 7. Epoch Train =====================================
print("\n=============== 开始训练 ===============")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    train_recon = model(train_x)
    train_loss = criterion(train_recon, train_x)
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_recon = model(val_x)
        val_loss = criterion(val_recon, val_x)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] | 训练损失: {train_loss.item():.6f} | 验证损失: {val_loss.item():.6f}")

# ===================================== 8. 提取核心特征【格式完美匹配：(n_samples, latent_dim)】=====================================
print("\n✅ 开始提取特征...")
model.eval()
with torch.no_grad():
    final_features = model.encode(data_tensor).numpy()

print(f"✅ 特征提取完成！特征形状: {final_features.shape}")  # 必输出 (n_samples, 8)

# ===================================== 9. 聚类+评估【原代码完全复用，一字未改】=====================================
scaler = StandardScaler()
features_scaled = scaler.fit_transform(final_features)

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(features_scaled)

sil_score = silhouette_score(features_scaled, cluster_labels)
print(f"\n=============== 聚类结果 ===============")
print(f"轮廓系数: {sil_score:.4f} (越接近1聚类效果越好)")
for label, count in zip(np.unique(cluster_labels), np.bincount(cluster_labels)):
    print(f"聚类 {label} → 样本数量: {count}")

# ===================================== 10. 保存结果【可选】=====================================
np.save("timesfm_final_features.npy", final_features)
np.save("cluster_labels.npy", cluster_labels)
print("\n✅ 结果已保存至本地！")
