import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# ===================== 0. 设备检查 =====================
TORCH_AVAILABLE = True
try:
    import torch
    print(f"✅ PyTorch版本: {torch.__version__}")
except (ImportError, OSError) as e:
    TORCH_AVAILABLE = False
    print(f"❌ 无法加载PyTorch (原因: {e})，将使用CPU替代方案")
    print(f"提示: 如果遇到 WinError 1114，通常是由于 PyTorch 环境问题，建议检查或重装 torch解决anaconda的data虚拟环境的报错")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")

# ===================== 1. 核心超参数 =====================
latent_dim = 64  # 提取的特征维度
epochs = 200
batch_size = 32
learning_rate = 0.001
selected_feature_idx = 0  # 选择第几个特征作为输入
patience = 5  # EarlyStopping 耐心值
do_visualize = True      # 是否生成并显示可视化图像

# ===================== 2. 加载并预处理数据 =====================
data_path = "./lstm_dbscan/data_washing_machine/data.npy"
seq_len_path = "./lstm_dbscan/data_washing_machine/seq_length.npy"

if not os.path.exists(data_path) or not os.path.exists(seq_len_path):
    raise FileNotFoundError("数据文件不存在，请检查路径。")

data = np.load(data_path)
seq_len = np.load(seq_len_path)

# 选择单个特征并保持 3D 形状 (samples, timesteps, features)
data_single_feature = data[:, :, selected_feature_idx:selected_feature_idx+1]

n_samples = data_single_feature.shape[0]
timesteps = data_single_feature.shape[1]
n_features = data_single_feature.shape[2]

print(f"数据加载完成 | 总样本数: {n_samples} | 填充后时间步: {timesteps} | 输入特征数: {n_features}")
print(f"样本真实长度范围: {np.min(seq_len)} ~ {np.max(seq_len)}")

# 数据归一化
X_min = data_single_feature.min()
X_max = data_single_feature.max()
X_scaled = (data_single_feature - X_min) / (X_max - X_min + 1e-7)

# 计算归一化后的 Mask 值
scaled_mask_value = (0.0 - X_min) / (X_max - X_min + 1e-7)
print(f"数据归一化完成 | 范围: {X_scaled.min():.2f} ~ {X_scaled.max():.2f} | 修正后的 Mask 值: {scaled_mask_value:.4f}")

# 转换为 PyTorch Tensor
X_tensor = torch.FloatTensor(X_scaled)

# ===================== 3. 构建 PyTorch LSTM 自编码器 =====================
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, timesteps):
        super(LSTMAutoencoder, self).__init__()
        self.timesteps = timesteps
        
        # 编码器
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.latent_layer = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
        # 解码器
        self.decoder_lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        # x: (batch, timesteps, input_dim)
        _, (h_n, _) = self.encoder_lstm(x)
        # h_n shape: (num_layers, batch, hidden_dim)
        latent = self.latent_layer(h_n[-1]) # 取最后一层的 hidden state
        return latent

    def forward(self, x):
        # 编码
        latent = self.encode(x) # (batch, latent_dim)
        
        # 解码准备 (RepeatVector)
        # 将 latent 复制 timesteps 次
        decoder_input = latent.unsqueeze(1).repeat(1, self.timesteps, 1) # (batch, timesteps, latent_dim)
        
        # 解码
        decoder_out, _ = self.decoder_lstm(decoder_input) # (batch, timesteps, hidden_dim)
        
        # 输出层 (TimeDistributed Dense)
        out = self.output_layer(decoder_out) # (batch, timesteps, input_dim)
        return out

model = LSTMAutoencoder(n_features, 32, latent_dim, timesteps).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ===================== 4. 准备训练 =====================
# 划分训练集和验证集
dataset = TensorDataset(X_tensor, X_tensor)
val_size = int(0.2 * n_samples)
train_size = n_samples - val_size
train_indices = list(range(train_size))
val_indices = list(range(train_size, n_samples))

train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False)

# Early Stopping 逻辑实现
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

early_stopping = EarlyStopping(patience=patience, verbose=True)

# ===================== 5. 训练循环 =====================
print("\n开始训练...")
for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    for batch_x, _ in train_loader:
        batch_x = batch_x.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        
        # 模拟 Masking：忽略填充值的损失计算
        mask = (batch_x != scaled_mask_value)
        loss = criterion(outputs[mask], batch_x[mask])
        
        loss.backward()
        # 梯度裁剪 (对应 tf Adam clipnorm=1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item() * batch_x.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, _ in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            
            mask = (batch_x != scaled_mask_value)
            loss = criterion(outputs[mask], batch_x[mask])
            
            val_loss += loss.item() * batch_x.size(0)
    
    val_loss /= len(val_loader.dataset)
    
    print(f'Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')
    
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("提前停止训练")
        break

# 加载最佳权重
model.load_state_dict(torch.load('checkpoint.pt'))

# ===================== 6. 提取时序特征 =====================
model.eval()
with torch.no_grad():
    X_tensor = X_tensor.to(device)
    X_lstm_extracted_features = model.encode(X_tensor).cpu().numpy()

# ===================== 结果输出 =====================
print(f"\n原始单特征时序数据形状: {X_scaled.shape}")
print(f"LSTM提取的特征形状: {X_lstm_extracted_features.shape}")
result_dir = "./lstm_dbscan/data_washing_machine/"

if do_visualize:
    print("\n正在生成 t-SNE 可视化...")
    
    # 对提取的特征进行 t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_lstm_extracted_features)
    
    plt.figure(figsize=(12, 6))
    
    # 子图 1: t-SNE 特征分布
    plt.subplot(1, 2, 1)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, c='blue', edgecolors='none', s=20)
    plt.title("t-SNE Visualization of LSTM Latent Features")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 子图 2: 原始数据抽样展示 (展示前10个样本)
    plt.subplot(1, 2, 2)
    for i in range(min(10, n_samples)):
        plt.plot(X_scaled[i, :seq_len[i], 0], alpha=0.7, label=f'Sample {i}')
    plt.title("Original Time Series Samples (Normalized)")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    # plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # 保存图像
    plot_path = f"{result_dir}tsne_distribution_torch.png"
    plt.savefig(plot_path, dpi=300)
    print(f"可视化图像已保存到: {plot_path}")
    
    # 显示图像
    plt.show()

# ===================== 保存结果 =====================

if not os.path.exists(result_dir):
    os.makedirs(result_dir)
np.save(f"{result_dir}lstm_ae_features.npy", X_lstm_extracted_features)
print(f"结果已保存到: {result_dir}lstm_ae_features.npy")

# 清理临时权重文件
if os.path.exists('checkpoint.pt'):
    os.remove('checkpoint.pt')
