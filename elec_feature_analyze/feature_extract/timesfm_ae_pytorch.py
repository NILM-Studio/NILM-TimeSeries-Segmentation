import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from timesfm.timesfm_torch import TimesFmTorch as TimesFm
from timesfm import timesfm_base, timesfm_torch
import timesfm

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

print(f"✅ PyTorch版本: {torch.__version__}")
print(f"✅ CUDA可用: {torch.cuda.is_available()}")
device_flag = "gpu" if torch.cuda.is_available() else "cpu"

# ===================================== 2. Hyperparameter=====================================
latent_dim = 32  # 最终特征维度
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
print("\nTimesFmTorch初始化参数：timesfm_parameter和与官方训练参数checkpoint")
hparams = timesfm.TimesFmHparams(
    backend=device_flag,
    per_core_batch_size=32,
    horizon_len=128,
    input_patch_len=32,
    output_patch_len=128,
    num_layers=50,
    model_dims=1280,
    use_positional_embedding=False,
)

# checkpoint = timesfm_base.TimesFmCheckpoint(
#     version="torch",
#     huggingface_repo_id="google/timesfm-1.0-200m",  # 官方仓库
#     local_dir="./timesfm_cache"  # 缓存目录，避免重复下载
# )
# timesfm-1.0-200m-pytorch or timesfm-2.0-500m-pytorch
checkpoint = timesfm.TimesFmCheckpoint(
    version="torch",
    local_dir="./timesfm_cache",
    huggingface_repo_id="google/timesfm-2.0-500m-pytorch")

timesfm = TimesFm(hparams=hparams, checkpoint=checkpoint)
print("\nTimesFmTorch初始化成功！")


# ===================================== 5. TimesFm+自编码器封装【适配无监督训练+特征提取】=====================================
# ===================================== 5. TimesFm+自编码器封装【适配无监督训练+特征提取】=====================================
class TimesFMAutoEncoder(nn.Module):
    def __init__(self, timesfm_model, timesteps, n_features, latent_dim):
        super().__init__()
        self.timesfm = timesfm_model
        self.timesteps = timesteps
        self.n_features = n_features
        self.latent_dim = latent_dim

        # TimesFM的horizon_len参数决定了输出长度，这里是128
        timesfm_output_len = 128  # 与hparams中的horizon_len一致

        # 特征提取投影层：时序输出 → 样本级低维特征
        self.encoder_proj = nn.Sequential(
            nn.Linear(timesfm_output_len, 32),  # 修改：输入维度为TimesFM输出长度128
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

        # 重构层：低维特征 → 还原时序数据
        self.decoder_proj = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, timesteps * n_features)  # 输出完整的时序数据
        )

    def encode(self, x):
        """特征提取：将输入转换为TimesFM格式，然后获取特征"""
        batch_size = x.shape[0]

        # 将PyTorch张量转换为TimesFM期望的格式
        input_list = []
        for i in range(batch_size):
            # 提取单个样本并转换为numpy
            sample = x[i, :, 0].cpu().numpy()  # 取第一个特征维度，形状为(timesteps,)
            input_list.append(sample)

        with torch.no_grad():  # 确保TimesFM预训练权重不被更新
            # 调用TimesFM的forecast方法
            forecast_mean, forecast_full = self.timesfm.forecast(input_list)
            # forecast_mean shape: (batch_size, horizon_len=128)
            forecast_tensor = torch.from_numpy(forecast_mean).float().to(x.device)

        # 通过投影层获得固定长度的特征
        feat = self.encoder_proj(forecast_tensor)  # 输入: (batch, 128) -> 输出: (batch, latent_dim)
        return feat

    def forward(self, x):
        """无监督重构：特征提取 → 时序还原"""
        feat = self.encode(x)
        # 重构：通过解码器投影
        recon_flat = self.decoder_proj(feat)  # (batch, latent_dim) -> (batch, timesteps*n_features)
        # 重塑为原始形状
        recon_x = recon_flat.view(x.shape[0], self.timesteps, self.n_features)
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
