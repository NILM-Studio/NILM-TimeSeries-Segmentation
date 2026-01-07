import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from fastdtw import fastdtw
from cluster_result_analyze import cluster_result_save
from sklearn.preprocessing import MinMaxScaler
import sys
import time

# 启用无缓冲输出，确保打印立即显示在日志中
sys.stdout.flush()
sys.stderr.flush()

BASE_DIR = r'./cluster_data/washing_machine_seg/'



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
    plt.show()
    if save_file is not None:
        plt.savefig(save_file)


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
    sys.stdout.flush()
    
    for i in range(len(data)-5000):
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

    # 1. 修复：自定义兼容标量的欧氏距离函数（解决ValueError）
    def scalar_euclidean(a, b):
        a = np.array(a)
        b = np.array(b)
        return np.linalg.norm(a - b)


    # 3. 手动计算DTW距离矩阵
    print(f"\n【DTW距离矩阵计算】")
    n = len(normalized_ts_list)
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
            dist, _ = fastdtw(normalized_ts_list[i], normalized_ts_list[j], dist=scalar_euclidean)
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

    # 4. DBSCAN聚类
    print(f"\n【DBSCAN聚类】")
    eps = 12
    min_pts = 2
    print(f"聚类参数: eps={eps}, min_samples={min_pts}")
    sys.stdout.flush()
    
    dbscan = DBSCAN(
        eps=eps,  # 对应DTW距离的实际数值范围（需观察矩阵调整）
        min_samples=min_pts,
        metric="precomputed"
    )
    print("开始DBSCAN聚类...")
    sys.stdout.flush()
    labels = dbscan.fit_predict(distance_matrix)

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
    
    print(f"\n【结果输出】")
    print(f"设备名称: {appliance_name}")
    print(f"结果保存路径: {result_save_path}")
    print(f"可视化结果路径: {visualize_save_path}")
    print(f"聚类结果分析目录: {cluster_result_dir}")
    sys.stdout.flush()
    
    # 可视化结果
    visualize_clusters(normalized_ts_list, labels, eps, min_pts, save_file=visualize_save_path)
    print(f"可视化结果已保存到: {visualize_save_path}")
    sys.stdout.flush()

    # 保存聚类结果
    np.save(result_save_path, labels)
    print(f"聚类结果已保存到: {result_save_path}")
    sys.stdout.flush()

    # 保存聚类分析结果
    cluster_result_save(normalized_ts_list, seq_len, labels, save_dir=cluster_result_dir)
    print(f"聚类分析结果已保存到: {cluster_result_dir}")
    sys.stdout.flush()

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)
    sys.stdout.flush()