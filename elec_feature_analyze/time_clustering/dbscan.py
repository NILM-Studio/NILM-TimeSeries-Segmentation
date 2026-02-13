import json
import os
import sys
import time
from datetime import datetime

import numpy as np
from fastdtw import fastdtw
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from tslearn.utils import to_time_series_dataset
from cluster_result_analyze import cluster_result_save, cluster_result_quantification
import matplotlib.pyplot as plt

# ======================== 常量配置（集中管理，方便修改） ========================
# 启用无缓冲输出，确保打印立即显示在日志中
sys.stdout.flush()
sys.stderr.flush()

BASE_DIR = r'./cluster_data/microwave'
DATA_PATH = os.path.join(BASE_DIR, 'data_fusion.npy')
FEATURES_PATH = os.path.join(BASE_DIR, 'bilstm_ae_features_cleaned_power_64_dim.npy')
SEQ_LEN_PATH = os.path.join(BASE_DIR, 'seq_length_fusion.npy')
DATA_MAPPING_FILE = os.path.join(BASE_DIR, 'data_mapping_fusion.json')
EXTERN_TAG = 'bilistm'     # 额外标签，在输出结果命名中添加额外的标签用于标识输出结果
CLUSTER_METHOD = 'dbscan'

CLUSTER_CONFIG = {
    'dbscan': {
        'method': 'dbscan',
        'eps': 0.4,
        'min_pts': 20,
        'metric': 'euclidean',
        'normalization_method': 'zscore'
    },
    'kmeans': {
        'method': 'kmeans',
        'n_neighbors': 5,
        'algorithm': 'auto',
        'metric': 'euclidean',
        'normalization_method': 'zscore'
    }
}


# ======================== 距离矩阵计算相关函数（原有逻辑保留） ========================
def compute_distance_matrix(data, metric='euclidean', metric_params=None):
    """
    计算距离矩阵

    参数：
        data (np.ndarray): 标准化后的特征数据 (n_sample, feature_dim)
        metric (str): 距离度量（同DBSCAN的metric参数）
        metric_params (dict): 距离度量的额外参数（如minkowski的p值）

    返回：
        distance_matrix (np.ndarray): 距离矩阵 (n_sample, n_sample)
    """
    print(f"计算{metric}距离矩阵...")
    if metric == 'dtw':
        return dtw_matrix_compute(data)
    elif metric == 'fastdtw':
        return fast_dtw_matrix_tslearn(data)
    else:
        # 处理度量参数（默认空字典）
        metric_params = metric_params or {}
        # 计算距离矩阵（cdist支持大部分常用距离）
        distance_matrix = cdist(data, data, metric=metric, **metric_params)
        print(f"距离矩阵形状: {distance_matrix.shape}")
        return distance_matrix


def fast_dtw_matrix_tslearn(ts_list):
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

    print("开始计算DTW距离矩阵...")
    sys.stdout.flush()

    # 并行计算DTW矩阵（底层优化）
    from tslearn.metrics import cdist_dtw
    dist_matrix = cdist_dtw(X, n_jobs=-1)  # n_jobs=-1并行

    elapsed_time = time.time() - start_time
    print(f"DTW距离矩阵计算完成！")
    print(f"矩阵形状: {dist_matrix.shape}")
    print(f"总耗时: {elapsed_time:.2f}秒")
    sys.stdout.flush()

    return dist_matrix


def dtw_matrix_compute(ts_list):
    """手动计算DTW距离矩阵（兼容标量的欧氏距离）"""

    # 自定义兼容标量的欧氏距离函数（解决ValueError）
    def scalar_euclidean(a, b):
        a = np.array(a)
        b = np.array(b)
        return np.linalg.norm(a - b)

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


# ======================== 数据加载与预处理 ========================
def load_data(data_path: str, feature_path: str, seq_len_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载特征数据和序列长度数据

    参数：
        data_path: 原始数据路径
        feature_path: 特征数据路径
        seq_len_path: 序列长度数据路径（seq_length.npy）

    返回：
        data_np: 原始特征数据
        seq_len: 序列长度数组
    """
    print(f"\n【数据源配置】")
    print(f"原始数据路径: {data_path}")
    print(f"特征数据路径: {feature_path}")
    print(f"序列长度文件路径: {seq_len_path}")

    # 加载数据
    data_np = np.load(data_path)
    seq_len = np.load(seq_len_path)
    
    # 检查特征路径是否存在以及是否有效
    feature_matrix = np.array([])  # 默认返回空数组
    if feature_path is not None and os.path.exists(feature_path):
        feature_matrix = np.load(feature_path)
        if feature_matrix.size == 0:
            print("警告: 特征数据文件存在但为空，将返回空的特征矩阵！")
            feature_matrix = np.array([])
        else:
            print(f"成功加载特征数据，特征矩阵大小: {feature_matrix.shape}")
    else:
        print(f"警告: 特征数据路径无效或不存在({feature_path})，将返回空的特征矩阵！")
        
    print(f"\n【数据加载】")
    print(f"加载数据成功，原始数据大小: {data_np.shape}")
    print(f"序列长度数组大小: {seq_len.shape}")
    print(f"特征矩阵大小: {feature_matrix.shape}")
    sys.stdout.flush()

    if data_np.size == 0:
        print("警告: 原始数据文件为空！")
        sys.stdout.flush()
        sys.exit(1)

    return data_np, feature_matrix, seq_len


def normalize_features(feature_matrix: np.ndarray, normalization_method: str = 'zscore') -> list[np.ndarray]:
    """
    特征归一化：全局归一化（专门针对特征矩阵）

    参数：
        feature_matrix: 特征矩阵，形状为 [样本数, 特征维度]
        normalization_method: 归一化方法，可选 'minmax' 或 'zscore'，默认为 'zscore'
            - 'minmax': Min-Max 归一化，将特征缩放到 [0, 1] 范围
            - 'zscore': Z-Score 标准化，将特征标准化为均值0、标准差1（推荐）

    返回：
        normalized_feature_list: 归一化后的特征列表
    """
    print(f"\n【特征归一化】")
    print(f"特征矩阵大小: {feature_matrix.shape if feature_matrix.size > 0 else 'Empty/None'}")
    print(f"归一化方法: {normalization_method}")
    sys.stdout.flush()

    # 检查特征矩阵是否为空
    if feature_matrix.size == 0:
        print("警告: 特征矩阵为空，无法进行特征聚类，退出程序")
        sys.exit(1)
    else:
        print("使用特征矩阵进行归一化")
        data = feature_matrix

    # 根据参数选择归一化方法
    if normalization_method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        print("使用 Min-Max 归一化（全局）")
    elif normalization_method == 'zscore':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        print("使用 Z-Score 标准化（全局）")
    else:
        raise ValueError(f"不支持的归一化方法: {normalization_method}，请选择 'minmax' 或 'zscore'")

    # 全局归一化：所有样本一起计算归一化参数
    normalized_features = scaler.fit_transform(data)

    # 转换为列表格式
    normalized_feature_list = [normalized_features[i] for i in range(len(normalized_features))]

    # 打印归一化统计信息
    print(f"归一化完成，有效序列数量: {len(normalized_feature_list)}")
    print(f"归一化后范围: [{normalized_features.min():.4f}, {normalized_features.max():.4f}]")
    print(f"归一化后均值: {normalized_features.mean():.4f}")
    print(f"归一化后标准差: {normalized_features.std():.4f}")
    sys.stdout.flush()
    
    return normalized_feature_list


# ======================== 距离矩阵缓存与获取 ========================
def get_distance_matrix(ts_list: list[np.ndarray], metric: str = 'euclidean') -> np.ndarray:
    """
    获取距离矩阵（直接计算）

    参数：
        ts_list: 时间序列列表
        metric: 距离度量方式

    返回：
        dist_matrix: 距离矩阵
    """
    print(f"开始计算{metric}距离矩阵...")
    sys.stdout.flush()
    
    # 直接计算距离矩阵
    dist_matrix = compute_distance_matrix(ts_list, metric)
    
    print(f"{metric}距离矩阵计算完成！")
    sys.stdout.flush()

    return dist_matrix


# ======================== DBSCAN聚类核心逻辑 ========================
def run_dbscan(dist_matrix: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    """
    执行DBSCAN聚类（基于预计算的距离矩阵）

    参数：
        dist_matrix: 预计算的距离矩阵
        eps: DBSCAN邻域半径
        min_pts: 最小样本数

    返回：
        labels: 聚类标签（-1表示噪声点）
    """
    print(f"\n【DBSCAN聚类】")
    print(f"聚类参数: eps={eps}, min_samples={min_pts}")
    sys.stdout.flush()

    dbscan_model = DBSCAN(
        eps=eps,
        min_samples=min_pts,
        metric="precomputed"
    )
    print("开始DBSCAN聚类...")
    sys.stdout.flush()
    labels = dbscan_model.fit_predict(dist_matrix)

    # 打印聚类基础结果
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    print(f"\n【聚类结果统计】")
    print(f"聚类数量: {n_clusters}")
    print(f"噪声点数量: {n_noise}")
    print(f"各类别样本数分布:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        label_name = "噪声点" if label == -1 else f"聚类 {label}"
        print(f"  {label_name}: {count} 个样本")
    sys.stdout.flush()

    return labels


# ======================== 结果评估与保存 ========================
def evaluate_clustering(labels: np.ndarray, dist_matrix: np.ndarray, org_data: np.ndarray, feature_matrix: np.ndarray,
                        save_dir: str, eps: float, min_pts: int) -> dict:
    """
    计算聚类评估指标并保存为JSON

    参数：
        labels: 聚类标签
        dist_matrix: 距离矩阵
        org_data: 原始数据
        feature_matrix: 特征数据
        save_dir: 评估指标保存目录

    返回：
        metrics: 评估指标字典
    """
    # 计算评估指标
    sil_score, db_score, ch_score = cluster_result_quantification(
        labels, dist_matrix, org_data, feature_matrix, save_dir
    )

    # 提取基础信息
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    appliance_name = BASE_DIR.split('/')[2]

    # 统计每个cluster的实例数量（包括噪声cluster）
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_distribution = {}
    for label, count in zip(unique_labels, counts):
        label_name = "noise" if label == -1 else f"cluster_{label}"
        cluster_distribution[label_name] = int(count)

    # 构造指标字典
    metrics = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "eps": str(eps),
        "min_pts": str(min_pts),
        "silhouette_score": str(sil_score),
        "davies_bouldin_score": str(db_score),
        "calinski_harabasz_score": str(ch_score),
        "appliance_name": appliance_name,
        "n_clusters": str(n_clusters),
        "n_noise": str(n_noise),
        "cluster_distribution": cluster_distribution
    }

    # 保存JSON
    json_save_path = os.path.join(save_dir, "evaluation_metrics.json")
    with open(json_save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n评估指标已保存到: {json_save_path}")
    sys.stdout.flush()

    return metrics


def save_clustering_results(
        data_np: np.ndarray,
        seq_len: np.ndarray,
        labels: np.ndarray,
        eps: float,
        min_pts: float
) -> str:
    """
    保存聚类结果（分析文件、可视化等），保存每一个簇的前50个原始数据图像，而非特征提取结果

    参数：
        data_np: 原始数据数组
        seq_len: 序列长度数组
        labels: 聚类标签
        eps: DBSCAN eps参数
        min_pts: DBSCAN min_pts参数

    返回：
        cluster_result_dir: 结果保存目录
    """
    appliance_name = BASE_DIR.split('/')[2]
    cluster_result_dir = rf'./cluster_data/dbscan_result/{appliance_name}/{eps}_{min_pts}_{EXTERN_TAG}/'
    labels_save_path = os.path.join(cluster_result_dir, f'cluster_labels.npy')

    # 创建目录
    os.makedirs(cluster_result_dir, exist_ok=True)

    # 循环遍历data_np的前len(labels)列，将其加入到data_list中
    data_list = []
    for i in range(min(len(labels), len(data_np))):
        data_list.append(data_np[i])
    
    # 保存聚类分析结果，传入data_list用于可视化
    cluster_result_save(data_list, seq_len, labels, save_dir=cluster_result_dir)
    
    # 保存labels数组
    np.save(labels_save_path, labels)
    print(f"聚类标签已保存到: {labels_save_path}")

    # 打印保存路径
    print(f"\n【结果输出路径】")
    print(f"设备名称: {appliance_name}")
    print(f"聚类结果文件: {labels_save_path}")
    print(f"聚类分析目录: {cluster_result_dir}")
    sys.stdout.flush()

    return cluster_result_dir


# ======================== EPS扫描函数 ========================
def scan_eps(data_np, features_matrix, config, save_dir=None,
             eps_start=0.1, eps_end=2.0, eps_step=0.1):
    """
    扫描不同的eps值，计算综合Loss，找到最优eps
    
    参数：
        data_np: 原始数据
        features_matrix: 特征矩阵
        config: 配置字典，包含以下键：
            min_pts: 固定的min_pts参数
            metric: 距离度量方法（可选）
            normalization_method: 特征归一化方法（可选）
        save_dir: 结果保存目录
        eps_start: eps起始值
        eps_end: eps结束值
        eps_step: eps步长
        
    返回：
        optimal_eps: 最优eps值
        eps_results: 所有eps的评估结果
    """
    # 从配置中获取参数
    min_pts = config['min_pts']
    metric = config.get('metric', 'euclidean')
    normalization_method = config.get('normalization_method', 'zscore')
    
    eps_values = np.arange(eps_start, eps_end + eps_step, eps_step)
    eps_results = []
    
    print(f"\n【EPS扫描】开始扫描 eps 范围: [{eps_start}, {eps_end}]，步长: {eps_step}")
    print(f"固定 min_pts: {min_pts}")
    
    # 1. 预计算归一化特征和距离矩阵（如果metric固定）
    normalized_feature_list = normalize_features(features_matrix, normalization_method=normalization_method)
    dist_matrix = get_distance_matrix(normalized_feature_list, metric=metric)
    
    # 2. 对每个eps执行聚类和评估
    for eps in eps_values:
        print(f"\n处理 eps = {eps:.2f}")
        
        # 执行DBSCAN聚类
        labels = run_dbscan(dist_matrix, eps, min_pts)
        
        # 计算评估指标
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
        
        # 调用现有的评估函数获取指标
        temp_save_dir = f'./temp_eps_scan/{eps:.2f}/'
        os.makedirs(temp_save_dir, exist_ok=True)
        
        sil_score, db_score, ch_score = cluster_result_quantification(
            labels, dist_matrix, data_np, features_matrix, temp_save_dir, visualize=False
        )
        
        # 计算转换后的Loss
        loss_sil = 1 - sil_score if sil_score is not None else 1.0
        loss_db = db_score if db_score is not None else 1.0
        loss_ch = 1 / (ch_score + 1e-6) if ch_score is not None else 1.0
        
        eps_results.append({
            'eps': eps,
            'sil_score': sil_score,
            'db_score': db_score,
            'ch_score': ch_score,
            'loss_sil': loss_sil,
            'loss_db': loss_db,
            'loss_ch': loss_ch,
            'n_clusters': n_clusters,
            'n_noise': n_noise
        })
    
    # 3. 归一化Loss并计算综合Loss
    if eps_results:
        # 提取所有Loss值
        loss_sil_values = [r['loss_sil'] for r in eps_results]
        loss_db_values = [r['loss_db'] for r in eps_results]
        loss_ch_values = [r['loss_ch'] for r in eps_results]
        
        # 计算每个Loss的最小值和最大值（用于归一化）
        min_loss_sil, max_loss_sil = min(loss_sil_values), max(loss_sil_values)
        min_loss_db, max_loss_db = min(loss_db_values), max(loss_db_values)
        min_loss_ch, max_loss_ch = min(loss_ch_values), max(loss_ch_values)
        
        # 计算综合Loss
        for r in eps_results:
            # 归一化
            r['loss_sil_norm'] = (r['loss_sil'] - min_loss_sil) / (max_loss_sil - min_loss_sil + 1e-6)
            r['loss_db_norm'] = (r['loss_db'] - min_loss_db) / (max_loss_db - min_loss_db + 1e-6)
            r['loss_ch_norm'] = (r['loss_ch'] - min_loss_ch) / (max_loss_ch - min_loss_ch + 1e-6)
            
            # 综合Loss（等权重）
            r['comprehensive_loss'] = (r['loss_sil_norm'] + r['loss_db_norm'] + r['loss_ch_norm']) / 3
    
    # 4. 找到最优eps（综合Loss最小）
    if eps_results:
        optimal_result = min(eps_results, key=lambda x: x['comprehensive_loss'])
        optimal_eps = optimal_result['eps']
        print(f"\n【最优EPS】找到最优 eps = {optimal_eps:.2f}")
        print(f"综合Loss: {optimal_result['comprehensive_loss']:.4f}")
        print(f"对应指标: 轮廓系数={optimal_result['sil_score']:.4f}, DBI={optimal_result['db_score']:.4f}, CHI={optimal_result['ch_score']:.4f}")
        print(f"簇数量: {optimal_result['n_clusters']}, 噪声点: {optimal_result['n_noise']}")
    else:
        optimal_eps = None
    
    # 5. 绘制Loss曲线
    plot_loss_curve(eps_results, save_dir)
    
    return optimal_eps, eps_results


def plot_loss_curve(eps_results, save_dir):
    """绘制综合Loss曲线，标记最优eps，同时显示cluster数量和噪声点数量
    
    参数：
        eps_results: eps扫描结果列表
        save_dir: 结果保存目录
    """
    if not eps_results:
        return
    
    eps_values = [r['eps'] for r in eps_results]
    comprehensive_losses = [r['comprehensive_loss'] for r in eps_results]
    n_clusters = [r['n_clusters'] for r in eps_results]
    n_noise = [r['n_noise'] for r in eps_results]
    
    # 找到最小Loss的索引
    min_loss_idx = np.argmin(comprehensive_losses)
    optimal_eps = eps_values[min_loss_idx]
    min_loss = comprehensive_losses[min_loss_idx]
    
    # 创建子图
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # 子图1: 综合Loss曲线
    axes[0].plot(eps_values, comprehensive_losses, 'b-o', label='Comprehensive Loss')
    axes[0].scatter([optimal_eps], [min_loss], color='r', s=100, label=f'Optimal eps={optimal_eps:.2f}')
    axes[0].set_title('Comprehensive Loss vs eps (DBSCAN)')
    axes[0].set_xlabel('eps')
    axes[0].set_ylabel('Comprehensive Loss')
    axes[0].grid(True)
    axes[0].legend()
    
    # 子图2: 簇数量随eps变化
    axes[1].plot(eps_values, n_clusters, 'g-o', label='Number of Clusters')
    axes[1].axvline(x=optimal_eps, color='r', linestyle='--', alpha=0.5, label=f'Optimal eps={optimal_eps:.2f}')
    axes[1].set_title('Number of Clusters vs eps')
    axes[1].set_xlabel('eps')
    axes[1].set_ylabel('Number of Clusters')
    axes[1].grid(True)
    axes[1].legend()
    
    # 子图3: 噪声点数量随eps变化
    axes[2].plot(eps_values, n_noise, 'm-o', label='Number of Noise Points')
    axes[2].axvline(x=optimal_eps, color='r', linestyle='--', alpha=0.5, label=f'Optimal eps={optimal_eps:.2f}')
    axes[2].set_title('Number of Noise Points vs eps')
    axes[2].set_xlabel('eps')
    axes[2].set_ylabel('Number of Noise Points')
    axes[2].grid(True)
    axes[2].legend()
    
    plt.tight_layout()
    
    # 保存图像
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'eps_scan_comprehensive.png'), dpi=300)
    plt.show()
    
    print(f"综合分析图已保存到: {os.path.join(save_dir, 'eps_scan_comprehensive.png')}")


# ======================== 主函数（串联所有流程） ========================
def main():
    """主函数：串联数据加载→预处理→距离矩阵→聚类→结果保存全流程"""
    # 1. 加载配置项
    config = CLUSTER_CONFIG[CLUSTER_METHOD]
    
    # 2. 加载数据
    data_np, features_matrix, seq_len = load_data(DATA_PATH, FEATURES_PATH, SEQ_LEN_PATH)
    
    if CLUSTER_METHOD == 'dbscan':
        # 检查是否需要扫描eps
        scan_eps_flag = True  # 设置为True启用扫描
        
        if scan_eps_flag:
            # 执行EPS扫描
            appliance_name = os.path.basename(BASE_DIR.rstrip(os.sep))
            save_dir = f'./cluster_data/dbscan_result/{appliance_name}/eps_scan/'
            optimal_eps, eps_results = scan_eps(
                data_np, features_matrix, 
                config=config,
                save_dir=save_dir,
                eps_start=0.02,
                eps_end=2.0,
                eps_step=0.02
            )
            
            # 保存扫描结果
            os.makedirs(save_dir, exist_ok=True)
            
            # 1. 先保存最佳的eps以及其相关结果
            optimal_result = next((r for r in eps_results if r['eps'] == optimal_eps), None)
            if optimal_result:
                optimal_eps_data = {
                    'optimal_eps': optimal_eps,
                    'optimal_result': optimal_result,
                    'scan_parameters': {
                        'min_pts': config['min_pts'],
                        'eps_start': 0.1,
                        'eps_end': 2.0,
                        'eps_step': 0.1,
                        'metric': config.get('metric', 'euclidean')
                    }
                }
                with open(os.path.join(save_dir, 'optimal_eps_result.json'), 'w', encoding='utf-8') as f:
                    json.dump(optimal_eps_data, f, ensure_ascii=False, indent=2)
                print(f"最佳EPS结果已保存到: {os.path.join(save_dir, 'optimal_eps_result.json')}")
            
            # 2. 再保存各个eps对应的结果
            with open(os.path.join(save_dir, 'eps_scan_results.json'), 'w', encoding='utf-8') as f:
                json.dump(eps_results, f, ensure_ascii=False, indent=2)
            print(f"所有EPS扫描结果已保存到: {os.path.join(save_dir, 'eps_scan_results.json')}")
        else:
            # 原有逻辑：使用固定eps
            eps, min_pts = config['eps'], config['min_pts']
            
            # 3. 特征归一化
            normalization_method = config.get('normalization_method', 'zscore')
            normalized_feature_list = normalize_features(features_matrix, normalization_method=normalization_method)
            
            # 4. 获取距离矩阵
            dist_matrix = get_distance_matrix(normalized_feature_list, metric=config['metric'])
            
            # 5. 执行DBSCAN聚类
            labels = run_dbscan(dist_matrix, eps, min_pts)
            
            # 6. 保存聚类结果
            cluster_result_dir = save_clustering_results(data_np, seq_len, labels, eps, min_pts)
            
            # 7. 评估聚类结果
            evaluate_clustering(labels, dist_matrix, data_np, features_matrix, cluster_result_dir, eps, min_pts)

    elif CLUSTER_METHOD == 'kmeans':
        pass

    # 9. 结束
    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
