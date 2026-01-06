import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset
from cluster_result_analyze import cluster_result_save

# # 1. 构造不等长时间序列（3类，每类5个样本，长度随机）
# rng = np.random.RandomState(42)
# noise_strength = 0.1
# n_samples_per_class = 5
# # 随机生成每类样本的长度（范围：50~100）
# sin_lengths = rng.randint(50, 100, size=n_samples_per_class)
# const_lengths = rng.randint(50, 100, size=n_samples_per_class)
# linear_lengths = rng.randint(50, 100, size=n_samples_per_class)
#
# # 类别1：不等长sin序列 + 扰动
# sin_samples = []
# for length in sin_lengths:
#     base = np.sin(np.linspace(0, 2*np.pi, length))
#     noisy = base + noise_strength * rng.randn(length)
#     sin_samples.append(noisy)
#
# # 类别2：不等长常数y=3序列 + 扰动
# const_samples = []
# for length in const_lengths:
#     base = np.ones(length) * 3
#     noisy = base + noise_strength * rng.randn(length)
#     const_samples.append(noisy)
#
# # 类别3：不等长线性y=x序列 + 扰动
# linear_samples = []
# for length in linear_lengths:
#     base = np.linspace(0, 5, length)
#     noisy = base + noise_strength * rng.randn(length)
#     linear_samples.append(noisy)
#
# # 拼接所有不等长样本
# X_unequal = sin_samples + const_samples + linear_samples
# print(f"不等长样本数量：{len(X_unequal)}")
# print(f"各样本长度：{[len(ts) for ts in X_unequal]}")  # 可看到长度不一致

# X_unequal = to_time_series_dataset(X_unequal)

X_unequal = np.load(r'./cluster_data/data.npy')
seq_length = np.load(r'./cluster_data/seq_length.npy')

# X_unequal = [data[i][:seq_length[i]].squeeze(-1) for i in range(len(data))]

# 2. 标准化（无需手动对齐长度）
X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X_unequal)

# 3. 训练模型（必须使用dtw/softdtw距离）
n_clusters = 10
ts_kmeans = TimeSeriesKMeans(
    n_clusters=n_clusters,
    metric="dtw",  # 关键：使用支持不等长的DTW距离
    max_iter=100,
    random_state=42
)
cluster_labels = ts_kmeans.fit_predict(X_scaled)

# 4. 输出结果
print(f"\n聚类标签：{cluster_labels}")
print(f"每类样本数：{np.bincount(cluster_labels)}")

# 5. 可视化（最长序列长度为基准，短序列已自动补零）
plt.figure(figsize=(12, 8))
for i in range(n_clusters):
    cluster_samples = X_scaled[cluster_labels == i]
    # 绘制当前类所有样本
    for sample in cluster_samples:
        plt.plot(sample[:, 0], color="lightgray", alpha=0.5)
    # 绘制聚类中心（长度=最长样本长度）
    plt.plot(ts_kmeans.cluster_centers_[i, :, 0], color=f"C{i}", linewidth=3, label=f"Cluster {i+1} Center")

plt.title("Unequal Length Time Series Clustering (DTW Metric)", fontsize=14)
plt.xlabel("Time Step (Aligned by Zero Padding)", fontsize=12)
plt.ylabel("Normalized Value", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.show()

np.save(r'./cluster_data/knn_cluster_result.npy', cluster_labels)
cluster_result_save(X_unequal, seq_length, cluster_labels, r'./cluster_data/knn_result/')
