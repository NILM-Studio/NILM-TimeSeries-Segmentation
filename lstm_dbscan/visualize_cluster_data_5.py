import json
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# 设置中文字体（解决Matplotlib中文乱码问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =============== 数据路径配置 =======================
DATA_DIR = r'f:\B__ProfessionProject\NILM\cluster_visualization\lstm_dbscan\data_dbscan'

CLUSTER_RESULT_FILE = os.path.join(DATA_DIR, 'detsec_clust_assignment.npy')

DATA_MAPPING_LIST = os.path.join(DATA_DIR, 'data_mapping_list.json')

ORIGINAL_DATA_FILE = os.path.join(DATA_DIR, 'data.npy')

FEATURES_FILE = os.path.join(DATA_DIR, 'detsec_features.npy')

# 没有序列长度文件时，默认使用 SEQ_LEN
SEQ_LEN_FILE = os.path.join(DATA_DIR, 'seq_length.npy')
SEQ_LEN = 64

OUTPUT_DIR = os.path.join(DATA_DIR, 'lstm_dbscan_visualizations')

# 图片样本数
N_SAMPLES = 50
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 时间单位 ==================
HOURS = 3600
DAYS = 24 * HOURS

def load_data(SEQ_LEN=64):
    """加载所有相关数据"""
    print(f"正在从 {DATA_DIR} 加载数据...")
    
    # 检查并加载各文件，如果不存在则返回 None
    cluster_labels = np.load(CLUSTER_RESULT_FILE) if os.path.exists(CLUSTER_RESULT_FILE) else None
    features = np.load(FEATURES_FILE) if os.path.exists(FEATURES_FILE) else None
    data = np.load(ORIGINAL_DATA_FILE) if os.path.exists(ORIGINAL_DATA_FILE) else None
    
    seq_lengths = np.load(SEQ_LEN_FILE) if os.path.exists(SEQ_LEN_FILE) else None
    if not isinstance(seq_lengths, np.ndarray) and data is not None:
        seq_lengths = np.full(len(data), SEQ_LEN)
    
    mapping_list = None
    if os.path.exists(DATA_MAPPING_LIST):
        with open(DATA_MAPPING_LIST, 'r', encoding='utf-8') as f:
            mapping_list = json.load(f)
        
    return cluster_labels, features, data, seq_lengths, mapping_list

def plot_tsne(features, labels, save_path=None):
    """绘制 t-SNE 降维分布图"""
    print("正在计算 t-SNE 降维...")
    
    # 如果输入是 3D (samples, seq_len, channels)，先展平为 2D (samples, seq_len * channels)
    if features.ndim == 3:
        n_samples = features.shape[0]
        features = features.reshape(n_samples, -1)
        print(f"  已将 3D 数据展平为 2D: {features.shape}")

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', n_jobs=-1)
    tsne_2d = tsne.fit_transform(features)
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_clusters)))
    
    plt.figure(figsize=(12, 8))
    for i, cluster_id in enumerate(unique_labels):
        idx = labels == cluster_id
        if cluster_id == -1:
            plt.scatter(tsne_2d[idx, 0], tsne_2d[idx, 1], c='black', marker='x', label='噪声点', alpha=0.5, s=30)
        else:
            plt.scatter(tsne_2d[idx, 0], tsne_2d[idx, 1], color=colors[i % 10], label=f'簇 {cluster_id}', alpha=0.7, s=50)
            
    plt.title('聚类结果 t-SNE 降维分布图', fontsize=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"t-SNE 图已保存至: {save_path}")


def plot_cluster_centers(data, labels, seq_lengths, save_path=None):
    """计算并绘制各簇的中心轮廓"""
    unique_labels = np.unique(labels)
    valid_labels = unique_labels[unique_labels != -1]
    n_clusters = len(valid_labels)
    
    if n_clusters == 0:
        print("未发现有效簇，跳过中心轮廓绘制。")
        return

    # 尝试导入 tslearn 用于高质量重心计算
    try:
        from tslearn.barycenters import dtw_barycenter_averaging
        use_dtw = True
        print("正在使用 DTW 重心计算 (Barycenter)...")
    except (ImportError, OSError) as e:
        use_dtw = False
        print(f"无法加载 tslearn (原因: {e})，将使用简单均值计算中心轮廓。")
        print("提示: 如果遇到 WinError 1114，通常是由于 PyTorch 环境问题，建议检查或重装 torch。")

    fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 3 * n_clusters), sharex=True)
    if n_clusters == 1:
        axes = [axes]
        
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_clusters)))
    
    for i, cluster_id in enumerate(valid_labels):
        cluster_idx = np.where(labels == cluster_id)[0]
        # 提取该簇的所有序列（只取第一通道数据进行中心计算）
        cluster_sequences = [data[idx, :seq_lengths[idx], 0] for idx in cluster_idx]
        
        if use_dtw:
            center = dtw_barycenter_averaging(cluster_sequences)
        else:
            # 简单对齐后求均值（处理不等长：取最短长度或填充）
            min_len = min(len(s) for s in cluster_sequences)
            center = np.mean([s[:min_len] for s in cluster_sequences], axis=0)
        
        # 绘制背景（所有样本）
        for seq in cluster_sequences:
            axes[i].plot(seq, color=colors[i % 10], alpha=0.1)
        
        # 绘制中心（加粗）
        axes[i].plot(center, color=colors[i % 10], linewidth=3, label=f'簇 {cluster_id} 中心')
        axes[i].set_title(f'簇 {cluster_id} 中心轮廓 (样本数: {len(cluster_idx)})')
        axes[i].legend(loc='upper right')
        axes[i].grid(alpha=0.3)
        axes[i].set_ylabel('功率 (W)')

    plt.xlabel('时间步')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"簇中心轮廓图已保存至: {save_path}")


def plot_time_distribution(mapping_list, labels, save_path=None):
    """可视化聚类结果按小时（0-23）的分布情况"""
    print("正在计算每小时时间分布...")
    
    # 初始化 24 个小时的统计 (0-23)
    n_bins = 24
    bin_labels = [f'{h:02d}:00' for h in range(n_bins)]
    
    unique_labels = np.unique(labels)
    cluster_stats = {cid: np.zeros(n_bins) for cid in unique_labels}
    
    for i, cid in enumerate(labels):
        start_ts = mapping_list[i]['start_timestamp']
        end_ts = mapping_list[i]['end_timestamp']
        
        # 获取开始时间的小时数
        dt = datetime.datetime.fromtimestamp(start_ts)
        hour = dt.hour
        
        # 累加持续时间
        duration = end_ts - start_ts
        if 0 <= hour < 24:
            cluster_stats[cid][hour] += duration
            
    # 绘图
    plt.figure(figsize=(12, 7))
    bottom = np.zeros(n_bins)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for i, cid in enumerate(unique_labels):
        label = f'簇 {cid}' if cid != -1 else '噪声'
        plt.bar(bin_labels, cluster_stats[cid], bottom=bottom, label=label, color=colors[i % 20])
        bottom += cluster_stats[cid]
        
    plt.title('聚类结果 24 小时分布统计', fontsize=15)
    plt.xlabel('时间 (小时)')
    plt.ylabel('运行总时长 (秒)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"每小时分布图已保存至: {save_path}")


def save_all_sequences_by_cluster(data, labels, seq_lengths, base_dir):
    """将每个簇随机抽取的样本波形图保存到对应的簇文件夹中"""
    print(f"正在按簇随机抽取最多 {N_SAMPLES} 个样本并保存波形图...")
    unique_labels = np.unique(labels)
    
    for cluster_id in unique_labels:
        # 创建文件夹
        cluster_folder_name = f'cluster_{cluster_id}' if cluster_id != -1 else 'cluster_noise'
        cluster_path = os.path.join(base_dir, cluster_folder_name)
        os.makedirs(cluster_path, exist_ok=True)
        
        # 找到该簇的所有索引
        all_indices = np.where(labels == cluster_id)[0]
        
        # 随机抽取样本（不放回）
        n_to_draw = min(len(all_indices), N_SAMPLES)
        sampled_indices = np.random.choice(all_indices, size=n_to_draw, replace=False)
        
        print(f"  正在保存 {cluster_folder_name} (从 {len(all_indices)} 个样本中随机抽取 {n_to_draw} 个)...")
        
        for idx in sampled_indices:
            # 提取数据并绘图
            sample_data = data[idx, :seq_lengths[idx], 0]
            plt.figure(figsize=(8, 4))
            plt.plot(sample_data)
            plt.title(f'Cluster {cluster_id} - Sample {idx}')
            plt.xlabel('Time Step')
            plt.ylabel('Power (W)')
            plt.grid(alpha=0.3)
            
            # 保存
            save_file = os.path.join(cluster_path, f'sample_{idx}.png')
            plt.savefig(save_file)
            plt.close()

def filter_small_clusters(labels, min_ratio=0.01):
    """
    检查聚类标签，对点数占比过小的簇进行剔除（标记为噪声 -1）
    
    参数:
    - labels: 聚类标签数组
    - min_ratio: 最小占比阈值，默认 0.01 (1%)。即样本数少于总样本数 1% 的簇将被剔除。
    """
    if labels is None:
        return None
    
    total_samples = len(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # 过滤掉已经是噪声的 -1，只处理有效的簇
    cluster_counts = {cid: count for cid, count in zip(unique_labels, counts) if cid != -1}
    
    # 如果没有有效簇，直接返回
    if not cluster_counts:
        return labels.copy()
    
    new_labels = labels.copy()
    removed_clusters = []
    
    # 计算每个簇的占比并检查是否低于阈值
    for cid, count in cluster_counts.items():
        ratio = count / total_samples
        if ratio < min_ratio:
            new_labels[new_labels == cid] = -1
            removed_clusters.append((cid, count, ratio))
            
    if removed_clusters:
        print(f"--- 异常值剔除报告 (阈值: {min_ratio*100:.1f}%) ---")
        for cid, count, ratio in removed_clusters:
            print(f"  簇 {cid} 因占比过小 ({ratio*100:.2f}% < {min_ratio*100:.1f}%, 样本数: {count}) 已被标记为噪声")
        print(f"----------------------")
    else:
        print(f"未发现占比小于 {min_ratio*100:.1f}% 的异常簇。")
        
    return new_labels

def main():
    try:
        # 1. 加载数据
        cluster_labels, features, data, seq_lengths, mapping_list = load_data(SEQ_LEN)
        
        # 检查核心数据是否加载成功
        if cluster_labels is None or data is None:
            print("错误: 无法加载核心数据（标签或原始数据不存在）。请检查路径。")
            return

        # 1.5 剔除点数过小的簇
        cluster_labels = filter_small_clusters(cluster_labels)

        # 2. 基础信息打印
        if features is not None:
            print(f"特征向量形状: {features.shape}")
        else:
            print("警告: 特征数据不存在。")

        n_samples = len(cluster_labels)
        n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
        n_noise = np.sum(cluster_labels == -1)
        print("-" * 40)
        print(f"样本总数: {n_samples}")
        print(f"聚类簇数: {n_clusters}")
        print(f"噪声点数: {n_noise}")
        print("-" * 40)
        
        # # 3. 可视化与保存
        # A. t-SNE 分布（features）
        if features is not None:
            plot_tsne(features, cluster_labels, save_path=os.path.join(OUTPUT_DIR, 'tsne_visualization.png'))

        # B. t-SNE 分布（data）
        if data is not None:
            plot_tsne(data, cluster_labels, save_path=os.path.join(OUTPUT_DIR, 'tsne_visualization_data.png'))
        
        # C. 簇中心轮廓
        if data is not None and seq_lengths is not None:
            plot_cluster_centers(data, cluster_labels, seq_lengths, save_path=os.path.join(OUTPUT_DIR, 'cluster_centers.png'))
        
        # D. 时间分布
        if mapping_list is not None:
            plot_time_distribution(mapping_list, cluster_labels, save_path=os.path.join(OUTPUT_DIR, 'time_distribution.png'))
        
        # E. 按簇保存所有序列波形图
        if data is not None and seq_lengths is not None:
            save_all_sequences_by_cluster(data, cluster_labels, seq_lengths, OUTPUT_DIR)
        
        print("\n所有可视化任务已完成！结果保存在:", OUTPUT_DIR)
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
