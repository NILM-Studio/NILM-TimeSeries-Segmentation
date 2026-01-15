import json
import os
import sys
import time
from datetime import datetime

import torch

import matplotlib.pyplot as plt
import numpy as np
from fastdtw import fastdtw
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset

from cluster_result_analyze import cluster_result_save, cluster_result_quantification

# 启用无缓冲输出，确保打印立即显示在日志中
sys.stdout.flush()
sys.stderr.flush()

BASE_DIR = r'./cluster_data/microwave/'


def visualize_clusters(time_series_list, labels, eps, min_pts, save_file=None, title="DBSCAN-DTW Clustering Results"):
    """
    可视化聚类结果（优化鲁棒性和可读性）

    Parameters:
    time_series_list: list of array-like
        时间序列列表
    labels: array-like
        聚类标签
    title: str
        图表标题
    """
    unique_labels = np.unique(labels)
    non_noise_labels = unique_labels[unique_labels != -1]
    n_non_noise = len(non_noise_labels)
    noise_indices = np.where(labels == -1)[0]

    # 处理全噪声场景
    if n_non_noise == 0:
        plt.figure(figsize=(12, 6))
        plt.title(f'{title} - All Noise Points')
        for idx in noise_indices:
            plt.plot(time_series_list[idx].flatten(), alpha=0.5, color='black',
                     label=f'Series {idx}' if idx < 5 else "")
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        return

    # 绘制聚类和噪声点
    plt.figure(figsize=(12, 2 * (n_non_noise + 1)))
    plt.suptitle(title, fontsize=14)

    # 绘制每个聚类
    for idx, cluster_id in enumerate(non_noise_labels):
        plt.subplot(n_non_noise + 1, 1, idx + 1)
        cluster_indices = np.where(labels == cluster_id)[0]
        for seq_idx in cluster_indices:
            plt.plot(time_series_list[seq_idx].flatten(), alpha=0.7, label=f'Series {seq_idx}')
        plt.title(f'Cluster {cluster_id} (n={len(cluster_indices)}) with eps={eps}, min_pts={min_pts}')
        plt.ylabel('Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(alpha=0.3)

    # 绘制噪声点
    if len(noise_indices) > 0:
        plt.subplot(n_non_noise + 1, 1, n_non_noise + 1)
        for seq_idx in noise_indices:
            plt.plot(time_series_list[seq_idx].flatten(), alpha=0.7, color='black', label=f'Series {seq_idx}')
        plt.title(f'Noise Points (n={len(noise_indices)})')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(alpha=0.3)

    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file)
    plt.show()
    plt.close()


def fast_dtw_matrix_tslearn(ts_list, filename):
    """
    使用tslearn库的cdist_dtw高效并行计算时间序列的DTW距离矩阵（底层C优化+多核并行）

    参数 (Inputs):
        ts_list : list of array-like (1D)

    返回值 (Outputs):
        dist_matrix : numpy.ndarray (2D, float)
            - 形状：(n, n)，其中n = len(ts_list)（序列数量）
            - 元素值：dist_matrix[i, j] 表示第i条和第j条序列的DTW距离
    """
    # 记录开始时间
    start_time = time.time()

    # 转换为tslearn格式
    X = to_time_series_dataset(ts_list)
    # 批量归一化
    # X = TimeSeriesScalerMeanVariance().fit_transform(X)

    print("开始计算DTW距离矩阵...")
    sys.stdout.flush()

    # 并行计算DTW矩阵（底层优化）
    from tslearn.metrics import cdist_dtw
    dist_matrix = cdist_dtw(X, n_jobs=-1)  # n_jobs=-1并行

    # 保存计算结果到缓存
    print(f"正在保存距离矩阵到缓存: {filename}")
    np.save(filename, dist_matrix)
    print(f"距离矩阵已保存到缓存文件: {filename}")

    elapsed_time = time.time() - start_time
    print(f"DTW距离矩阵计算完成！")
    print(f"矩阵形状: {dist_matrix.shape}")
    print(f"总耗时: {elapsed_time:.2f}秒")
    sys.stdout.flush()

    return dist_matrix


def dtw_matrix_compute(ts_list):
    """

    :param ts_list:
    :return:
    """

    # 自定义兼容标量的欧氏距离函数（解决ValueError）
    def scalar_euclidean(a, b):
        a = np.array(a)
        b = np.array(b)
        return np.linalg.norm(a - b)

    # 手动计算DTW距离矩阵
    print(f"\n【DTW距离矩阵计算】")
    n = len(ts_list)
    distance_matrix = np.zeros((n, n))

    # 计算总需要计算的配对数（上三角矩阵）
    total_pairs = n * (n - 1) // 2
    processed_pairs = 0
    print(f"序列数量: {n}")
    print(f"需要计算的距离对数量: {total_pairs}")
    print(f"预计计算时间: 约 {total_pairs * 0.01:.2f} 秒")
    sys.stdout.flush()

    # 性能优化：添加时间监控
    start_time = time.time()

    for i in range(n):
        for j in range(i + 1, n):
            # 使用自定义距离函数，避免报错
            dist, _ = fastdtw(ts_list[i], ts_list[j], dist=scalar_euclidean)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # 距离矩阵对称

            # 更新进度
            processed_pairs += 1
            if processed_pairs % 100 == 0 or processed_pairs == total_pairs:  # 每100个或完成时打印一次
                progress_percent = (processed_pairs / total_pairs) * 100
                elapsed_time = time.time() - start_time
                print(f"  进度: {progress_percent:.2f}% ({processed_pairs}/{total_pairs}) - 耗时: {elapsed_time:.2f}秒")
                sys.stdout.flush()

    print("DTW距离矩阵计算完成！")
    print(f"距离矩阵大小: {distance_matrix.shape}")
    print(f"总耗时: {time.time() - start_time:.2f}秒")
    sys.stdout.flush()
    return distance_matrix


def gpu_dtw_matrix_torch(ts_list, device="cuda:0"):
    """
    PyTorch GPU加速DTW距离矩阵（批量并行，性能最优）
    :param ts_list: 时间序列列表
    :param device: GPU设备（默认cuda:0）
    :return: 距离矩阵（CPU numpy数组）
    """
    # 1. 数据预处理
    X = to_time_series_dataset(ts_list)
    n = len(X)

    # 2. 转移到GPU
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    # 统一序列长度（补0，不影响DTW结果）
    max_len = max([len(ts) for ts in X])
    X_padded = torch.zeros((n, max_len), dtype=torch.float32).to(device)
    for i in range(n):
        X_padded[i, :len(X[i])] = torch.tensor(X[i].flatten(), dtype=torch.float32).to(device)

    # 3. 批量DTW计算（向量化+GPU并行）
    dist_matrix = torch.zeros((n, n), dtype=torch.float32).to(device)
    for i in range(n):
        # 单次计算i与所有j的DTW（广播加速）
        ts_i = X_padded[i:i + 1, :len(X[i])]
        for j in range(i, n):
            ts_j = X_padded[j:j + 1, :len(X[j])]

            # PyTorch版DTW（优化的动态规划）
            len1, len2 = ts_i.shape[1], ts_j.shape[1]
            cost = torch.zeros((len1 + 1, len2 + 1), dtype=torch.float32).to(device)
            cost[:, 0] = float('inf')
            cost[0, :] = float('inf')
            cost[0, 0] = 0.0

            for a in range(1, len1 + 1):
                for b in range(1, len2 + 1):
                    cost[a, b] = torch.abs(ts_i[0, a - 1] - ts_j[0, b - 1]) + torch.min(
                        torch.stack([cost[a - 1, b], cost[a, b - 1], cost[a - 1, b - 1]])
                    )

            dist_matrix[i, j] = cost[len1, len2]
            dist_matrix[j, i] = cost[len1, len2]

    # 4. GPU→CPU
    return dist_matrix.cpu().numpy()


def dbscan_dtw(ts_list, eps, min_pts):
    """

    :param ts_list:
    :param eps:
    :param min_pts:
    :return:
    """
    n = len(ts_list)
    dtw_matrix_filename = os.path.join(BASE_DIR, f'dist_matrix_{n}x{n}.npy')

    # 检查是否存在缓存文件
    if os.path.exists(dtw_matrix_filename):
        print(f"发现缓存文件: {dtw_matrix_filename}，正在加载...")
        dist_matrix = np.load(dtw_matrix_filename)
        print(f"使用缓存，节省计算时间: 已跳过DTW距离矩阵计算")
        sys.stdout.flush()
    else:
        print(f"未发现缓存文件: {dtw_matrix_filename}，计算DTW距离矩阵...")
        dist_matrix = fast_dtw_matrix_tslearn(ts_list, dtw_matrix_filename)
    print(f"\n【DBSCAN聚类】")
    print(f"聚类参数: eps={eps}, min_samples={min_pts}")
    sys.stdout.flush()
    dbscan = DBSCAN(
        eps=eps,  # 对应DTW距离的实际数值范围（需观察矩阵调整）
        min_samples=min_pts,
        metric="precomputed"
    )
    print("开始DBSCAN聚类...")
    sys.stdout.flush()
    labels = dbscan.fit_predict(dist_matrix)
    return labels, dist_matrix


# 示例用法
if __name__ == "__main__":
    # 打印当前配置信息，确保立即输出
    print("=" * 60)
    print("DBSCAN Time Series Clustering")
    print("=" * 60)
    sys.stdout.flush()

    # 数据源信息
    data_path = BASE_DIR + 'data.npy'
    seq_len_path = BASE_DIR + 'seq_length.npy'
    print(f"\n【数据源配置】")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"数据文件路径: {data_path}")
    print(f"序列长度文件路径: {seq_len_path}")
    sys.stdout.flush()

    # 加载数据
    data_np = np.load(data_path)
    print(f"\n【数据加载】")
    print(f"加载数据成功，数据大小: {data_np.shape}")
    sys.stdout.flush()

    if data_np.size == 0:
        print("警告: data.npy 是空文件")
        sys.stdout.flush()
        exit()
    data = data_np[:, :, 0]
    seq_len = np.load(seq_len_path)
    time_series_data = []
    normalized_ts_list = []

    # 遍历data每一行，将i行的前seq_len[i]个数提取出来作为序列append到data_list中
    print(f"\n【数据预处理】")
    print(f"原始数据行数量: {len(data)}")
    print(f"序列长度数组大小: {len(seq_len)}")
    data = data[:300]
    sys.stdout.flush()

    for i in range(len(data)):
        if seq_len[i] == 0:
            continue
        sequence = data[i][:seq_len[i]]  # 提取第i行的前seq_len[i]个数
        time_series_data.append(sequence)

        # 对当前序列进行归一化处理
        min_max_scaler = MinMaxScaler()
        # 将序列reshape为(n, 1)格式以适应MinMaxScaler
        sequence_reshaped = sequence.reshape(-1, 1)
        normalized_sequence = min_max_scaler.fit_transform(sequence_reshaped).flatten()
        normalized_ts_list.append(normalized_sequence)

    print(f"处理完成，有效序列数量: {len(normalized_ts_list)}")
    sys.stdout.flush()

    eps, min_pts = 0.9, 2
    labels, dist_matrix = dbscan_dtw(normalized_ts_list, eps, min_pts)

    # 5. 输出结果
    print(f"\n【聚类结果】")
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    print(f"聚类数量: {n_clusters}")
    print(f"噪声点数量: {n_noise}")
    print(f"各类别样本数分布:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        label_name = "噪声点" if label == -1 else f"聚类 {label}"
        print(f"  {label_name}: {count} 个样本")
    sys.stdout.flush()

    # 输出文件配置
    appliance_name = BASE_DIR.split('/')[2]
    result_save_path = BASE_DIR + f'dbscan_result_{eps}_{min_pts}.npy'
    visualize_save_path = f'./cluster_data/dbscan_result/{appliance_name}/{eps}_{min_pts}/dbscan_result_{eps}_{min_pts}.png'
    cluster_result_dir = rf'./cluster_data/dbscan_result/{appliance_name}/{eps}_{min_pts}/'

    sil_score, db_score, ch_score = cluster_result_quantification(labels, dist_matrix, data, f'./cluster_data/dbscan_result/{appliance_name}/{eps}_{min_pts}/')

    # 保存聚类评估指标到JSON文件
    evaluation_metrics = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "eps": str(eps),
        "min_pts": str(min_pts),
        "silhouette_score": str(sil_score),
        "davies_bouldin_score": str(db_score),
        "calinski_harabasz_score": str(ch_score),
        "appliance_name": appliance_name,
        "n_clusters": str(n_clusters),
        "n_noise": str(n_noise)
    }

    # 确保目录存在
    os.makedirs(cluster_result_dir, exist_ok=True)

    # 保存JSON文件
    json_save_path = os.path.join(cluster_result_dir, "evaluation_metrics.json")
    with open(json_save_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_metrics, f, ensure_ascii=False, indent=2)

    print(f"\n【结果输出】")
    print(f"设备名称: {appliance_name}")
    print(f"结果保存路径: {result_save_path}")
    print(f"可视化结果路径: {visualize_save_path}")
    print(f"聚类结果分析目录: {cluster_result_dir}")
    sys.stdout.flush()

    # 保存聚类分析结果
    cluster_result_save(normalized_ts_list, seq_len, labels, save_dir=cluster_result_dir)
    print(f"聚类分析结果已保存到: {cluster_result_dir}")
    sys.stdout.flush()

    # 保存聚类结果
    np.save(result_save_path, labels)
    print(f"聚类结果已保存到: {result_save_path}")
    sys.stdout.flush()

    # 可视化结果
    visualize_clusters(normalized_ts_list, labels, eps, min_pts, save_file=visualize_save_path)
    print(f"可视化结果已保存到: {visualize_save_path}")
    sys.stdout.flush()

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)
    sys.stdout.flush()
