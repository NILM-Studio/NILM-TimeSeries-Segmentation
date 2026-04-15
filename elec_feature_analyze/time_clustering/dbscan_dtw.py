#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DTW-DBSCAN 聚类工具

功能：
1. 加载原始数据序列 data.npy (x,y,z)
2. 指定列 k_index 将其转换为 (x,y) 形状
3. 使用 tslearn 库计算 DTW 距离矩阵
4. 保存距离矩阵到 BASE_DIR
5. 实现基于 DTW 距离矩阵的 DBSCAN 聚类
6. 支持 EPS 扫描功能，寻找最优参数
7. 使用 SCI, CHI 和 DBI 评估聚类结果
8. 保存评估结果为 JSON 文件

使用方法：
    修改文件开头的配置参数后运行：python dtw_dbscan.py
"""
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tslearn.utils import to_time_series_dataset
from tslearn.metrics import cdist_dtw
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt

# 导入聚类结果分析模块
from cluster_result_analyze import cluster_result_pic_save, cluster_result_quantification

# ======================== 常量配置（集中管理，方便修改） ========================
# 启用无缓冲输出，确保打印立即显示在日志中
# 在 Slurm 环境中，建议使用 -u 参数运行: python -u dtw_dbscan.py
# 或者在 Slurm 脚本中设置: export PYTHONUNBUFFERED=1

APPLIANCE_NAME = 'washing_machine'
# 获取当前脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, 'cluster_data_hpc', APPLIANCE_NAME)
DATA_FILE = os.path.join(BASE_DIR, 'data_fusion.npy')  # 原始数据文件 (x,y,z)
SEQ_LEN_PATH = os.path.join(BASE_DIR, 'seq_length_fusion.npy')  # 序列长度文件
K_INDEX = 2  # 要使用的特征列索引（0-based）
EXTERNAL_TAG = 'dtw'  # 外部指标标签



print(f"设备名称: {APPLIANCE_NAME}", flush=True)

# DBSCAN 配置
CLUSTER_CONFIG = {
    'dbscan': {
        'method': 'dbscan',
        'min_pts': 20,
        'eps': 1.50,
        'normalization_method': 'zscore'
    }
}

# EPS 扫描配置
SCAN_OPT_EPS = False  # 是否启用 EPS 扫描
EPS_START = 0.02  # EPS 起始值
EPS_END = 2.0    # EPS 结束值
EPS_STEP = 0.02  # EPS 步长

# 结果保存目录
if SCAN_OPT_EPS:
    SAVE_DIR = os.path.join(SCRIPT_DIR, 'cluster_data_hpc', 'dbscan_result', APPLIANCE_NAME, 'eps_scan_dtw')
else:
    SAVE_DIR = os.path.join(SCRIPT_DIR, 'cluster_data_hpc', 'dbscan_result', APPLIANCE_NAME, f"{CLUSTER_CONFIG['dbscan']['eps']}_{CLUSTER_CONFIG['dbscan']['min_pts']}_dtw")

# 缓存配置
USE_CACHE = True  # 是否使用 DTW 距离矩阵缓存

# 创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"结果保存目录: {SAVE_DIR}")

# ======================== 数据加载与预处理 ========================
def load_and_preprocess_data(data_file, seq_len_path, k_index):
    """
    加载数据并进行预处理

    参数：
        data_file: 数据文件路径
        seq_len_path: 序列长度文件路径
        k_index: 要使用的特征列索引

    返回：
        data_2d: 处理后的数据 (n_samples, max_seq_len)
        data_original: 原始数据（用于后续处理）
        seq_len: 序列长度数组
    """
    print(f"\n【数据加载与预处理】", flush=True)
    print(f"加载数据文件: {data_file}", flush=True)
    print(f"加载序列长度文件: {seq_len_path}", flush=True)
    
    # 加载数据
    data_original = np.load(data_file)
    print(f"原始数据形状: {data_original.shape}", flush=True)
    
    # 加载序列长度
    seq_len = np.load(seq_len_path)
    print(f"序列长度数组形状: {seq_len.shape}", flush=True)
    
    # 检查数据维度
    if data_original.ndim != 3:
        print(f"警告: 数据维度不是 3D，当前维度: {data_original.ndim}")
        sys.exit(1)
    
    # 检查 k_index 是否有效
    if k_index < 0 or k_index >= data_original.shape[2]:
        print(f"错误: 特征列索引 {k_index} 超出范围 [0, {data_original.shape[2]-1}]")
        sys.exit(1)
    
    # 提取指定列，转换为 (n_samples, max_seq_len) 形状
    data_2d = data_original[:, :, k_index]
    print(f"提取后的数据形状: {data_2d.shape}")
    
    return data_2d, data_original, seq_len


def normalize_data(data, normalization_method='zscore'):
    """
    数据归一化

    参数：
        data: 输入数据 (n_samples, max_seq_len)
        normalization_method: 归一化方法

    返回：
        normalized_data: 归一化后的数据
    """
    print(f"\n【数据归一化】")
    print(f"归一化方法: {normalization_method}")
    
    # 检查数据维度
    if data.ndim != 2:
        print(f"错误: 数据维度不是 2D，当前维度: {data.ndim}")
        sys.exit(1)
    
    # 全局归一化
    if normalization_method == 'zscore':
        scaler = StandardScaler()
        print("使用 Z-Score 标准化（全局）")
    else:
        print(f"警告: 不支持的归一化方法: {normalization_method}，使用 Z-Score")
        scaler = StandardScaler()
    
    # 对每个样本进行归一化
    normalized_data = []
    for i in range(data.shape[0]):
        sample = data[i].reshape(-1, 1)
        normalized_sample = scaler.fit_transform(sample).flatten()
        normalized_data.append(normalized_sample)
    
    print(f"归一化完成，样本数量: {len(normalized_data)}")
    return normalized_data

# ======================== DTW 距离矩阵计算 ========================
def compute_dtw_distance_matrix(data):
    """
    计算 DTW 距离矩阵并保存（支持缓存）

    参数：
        data: 归一化后的数据列表

    返回：
        dist_matrix: DTW 距离矩阵
    """
    print(f"\n【计算 DTW 距离矩阵】", flush=True)
    
    # 转换为 tslearn 格式
    X = to_time_series_dataset(data)
    print(f"数据转换为 tslearn 格式，形状: {X.shape}", flush=True)
    
    # 生成缓存文件名
    cache_path = os.path.join(BASE_DIR, f"dtw_dist_matrix.npy")
    
    # 检查缓存是否存在且启用缓存
    if USE_CACHE and os.path.exists(cache_path):
        print(f"加载缓存的距离矩阵: {cache_path}", flush=True)
        dist_matrix = np.load(cache_path)
        print(f"矩阵形状: {dist_matrix.shape}", flush=True)
        return dist_matrix
    
    # 记录开始时间
    start_time = time.time()
    
    # 并行计算 DTW 距离矩阵
    print("开始计算 DTW 距离矩阵...", flush=True)
    dist_matrix = cdist_dtw(X, n_jobs=-1)  # n_jobs=-1 启用并行
    
    elapsed_time = time.time() - start_time
    print(f"DTW 距离矩阵计算完成！", flush=True)
    print(f"矩阵形状: {dist_matrix.shape}", flush=True)
    print(f"总耗时: {elapsed_time:.2f}秒", flush=True)
    
    # 保存距离矩阵（缓存）
    if USE_CACHE:
        np.save(cache_path, dist_matrix)
        print(f"距离矩阵已保存到: {cache_path}", flush=True)
    
    return dist_matrix

# ======================== DBSCAN 聚类 ========================
def run_dbscan(dist_matrix, eps, min_pts):
    """
    执行 DBSCAN 聚类

    参数：
        dist_matrix: 距离矩阵
        eps: DBSCAN 邻域半径
        min_pts: 最小样本数

    返回：
        labels: 聚类标签
    """
    print(f"\n【DBSCAN 聚类】")
    print(f"聚类参数: eps={eps}, min_samples={min_pts}")
    
    dbscan_model = DBSCAN(
        eps=eps,
        min_samples=min_pts,
        metric="precomputed"
    )
    
    print("开始 DBSCAN 聚类...")
    labels = dbscan_model.fit_predict(dist_matrix)
    
    # 打印聚类结果
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    print(f"聚类数量: {n_clusters}")
    print(f"噪声点数量: {n_noise}")
    
    return labels

# ======================== 聚类评估 ========================
def evaluate_clustering(labels, dist_matrix):
    """
    评估聚类结果

    参数：
        labels: 聚类标签
        dist_matrix: 距离矩阵

    返回：
        metrics: 评估指标字典
    """
    print(f"\n【评估聚类结果】")
    
    metrics = {}
    
    # 计算轮廓系数 (SCI)
    try:
        if len(set(labels)) > 1:
            sil_score = silhouette_score(dist_matrix, labels, metric='precomputed')
            metrics['sil_score'] = sil_score
            print(f"轮廓系数 (SCI): {sil_score:.4f}")
        else:
            metrics['sil_score'] = None
            print("警告: 聚类数量不足，无法计算轮廓系数")
    except Exception as e:
        metrics['sil_score'] = None
        print(f"计算轮廓系数时出错: {e}")
    
    # 计算 Davies-Bouldin 指数 (DBI)
    try:
        if len(set(labels)) > 1:
            db_score = davies_bouldin_score(dist_matrix, labels)
            metrics['db_score'] = db_score
            print(f"Davies-Bouldin 指数 (DBI): {db_score:.4f}")
        else:
            metrics['db_score'] = None
            print("警告: 聚类数量不足，无法计算 Davies-Bouldin 指数")
    except Exception as e:
        metrics['db_score'] = None
        print(f"计算 Davies-Bouldin 指数时出错: {e}")
    
    # 计算 Calinski-Harabasz 指数 (CHI)
    try:
        if len(set(labels)) > 1:
            ch_score = calinski_harabasz_score(dist_matrix, labels)
            metrics['ch_score'] = ch_score
            print(f"Calinski-Harabasz 指数 (CHI): {ch_score:.4f}")
        else:
            metrics['ch_score'] = None
            print("警告: 聚类数量不足，无法计算 Calinski-Harabasz 指数")
    except Exception as e:
        metrics['ch_score'] = None
        print(f"计算 Calinski-Harabasz 指数时出错: {e}")
    
    # 基础统计信息
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))  # 转换为 Python int
    metrics['n_clusters'] = int(n_clusters)  # 转换为 Python int
    metrics['n_noise'] = int(n_noise)  # 转换为 Python int
    
    return metrics

# ======================== EPS 扫描 ========================
def scan_eps(dist_matrix, config, eps_start, eps_end, eps_step):
    """
    扫描不同的 eps 值，评估聚类结果

    参数：
        dist_matrix: 距离矩阵
        config: 配置字典
        eps_start: EPS 起始值
        eps_end: EPS 结束值
        eps_step: EPS 步长

    返回：
        optimal_eps: 最优 EPS 值
        eps_results: 所有 EPS 的评估结果
    """
    print(f"\n【EPS 扫描】", flush=True)
    print(f"扫描范围: [{eps_start}, {eps_end}]，步长: {eps_step}", flush=True)
    
    min_pts = config['min_pts']
    eps_values = np.arange(eps_start, eps_end + eps_step, eps_step)
    eps_results = []
    
    for eps in eps_values:
        print(f"\n处理 eps = {eps:.2f}", flush=True)
        
        # 执行 DBSCAN 聚类
        labels = run_dbscan(dist_matrix, eps, min_pts)
        
        # 评估聚类结果
        metrics = evaluate_clustering(labels, dist_matrix)
        
        # 记录结果
        result = {
            'eps': eps,
            'sil_score': metrics.get('sil_score'),
            'db_score': metrics.get('db_score'),
            'ch_score': metrics.get('ch_score'),
            'n_clusters': metrics.get('n_clusters'),
            'n_noise': metrics.get('n_noise')
        }
        eps_results.append(result)
    
    # 保存扫描结果
    save_path = os.path.join(SAVE_DIR, 'eps_scan_results.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(eps_results, f, ensure_ascii=False, indent=2)
    print(f"\n扫描结果已保存到: {save_path}")
    
    # 找到最优 EPS（基于轮廓系数）
    optimal_eps = None
    if eps_results:
        # 过滤掉无法计算轮廓系数的结果
        valid_results = [r for r in eps_results if r['sil_score'] is not None]
        if valid_results:
            # 选择轮廓系数最大的结果
            optimal_result = max(valid_results, key=lambda x: x['sil_score'])
            optimal_eps = optimal_result['eps']
            print(f"\n【最优 EPS】")
            print(f"最优 eps: {optimal_eps:.2f}")
            print(f"轮廓系数: {optimal_result['sil_score']:.4f}")
            print(f"簇数量: {optimal_result['n_clusters']}")
            print(f"噪声点数量: {optimal_result['n_noise']}")
    
    # 可视化 EPS 扫描结果
    if eps_results:
        print(f"\n【可视化 EPS 扫描结果】")
        
        # 提取数据
        eps_list = [r['eps'] for r in eps_results]
        sci_list = [r['sil_score'] if r['sil_score'] is not None else 0 for r in eps_results]
        dbi_list = [r['db_score'] if r['db_score'] is not None else 0 for r in eps_results]
        chi_list = [r['ch_score'] if r['ch_score'] is not None else 0 for r in eps_results]
        n_clusters_list = [r['n_clusters'] for r in eps_results]
        n_noise_list = [r['n_noise'] for r in eps_results]
        
        # 创建画布
        plt.figure(figsize=(15, 12))
        
        # 1. 轮廓系数 (SCI)
        plt.subplot(3, 2, 1)
        plt.plot(eps_list, sci_list, 'b-o', markersize=4)
        plt.title('Silhouette Score (SCI) vs EPS')
        plt.xlabel('EPS')
        plt.ylabel('SCI')
        plt.grid(True, alpha=0.3)
        
        # 2. Davies-Bouldin 指数 (DBI)
        plt.subplot(3, 2, 2)
        plt.plot(eps_list, dbi_list, 'r-o', markersize=4)
        plt.title('Davies-Bouldin Index (DBI) vs EPS')
        plt.xlabel('EPS')
        plt.ylabel('DBI')
        plt.grid(True, alpha=0.3)
        
        # 3. Calinski-Harabasz 指数 (CHI)
        plt.subplot(3, 2, 3)
        plt.plot(eps_list, chi_list, 'g-o', markersize=4)
        plt.title('Calinski-Harabasz Index (CHI) vs EPS')
        plt.xlabel('EPS')
        plt.ylabel('CHI')
        plt.grid(True, alpha=0.3)
        
        # 4. 簇数量
        plt.subplot(3, 2, 4)
        plt.plot(eps_list, n_clusters_list, 'c-o', markersize=4)
        plt.title('Number of Clusters vs EPS')
        plt.xlabel('EPS')
        plt.ylabel('Number of Clusters')
        plt.grid(True, alpha=0.3)
        
        # 5. 噪声数量
        plt.subplot(3, 2, 5)
        plt.plot(eps_list, n_noise_list, 'm-o', markersize=4)
        plt.title('Number of Noise Points vs EPS')
        plt.xlabel('EPS')
        plt.ylabel('Number of Noise Points')
        plt.grid(True, alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(SAVE_DIR, 'eps_scan_visualization.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
        
        # 关闭图像
        plt.close()
    
    return optimal_eps, eps_results

# ======================== 主函数 ========================
def main():
    """
    主函数
    """
    print("=" * 60, flush=True)
    print("DTW-DBSCAN 聚类工具", flush=True)
    print("=" * 60, flush=True)
    
    # 1. 加载和预处理数据
    data_2d, data_original, seq_len = load_and_preprocess_data(DATA_FILE, SEQ_LEN_PATH, K_INDEX)
    
    # 2. 数据归一化
    normalized_data = normalize_data(data_2d, CLUSTER_CONFIG['dbscan']['normalization_method'])
    
    # 3. 计算 DTW 距离矩阵
    dist_matrix = compute_dtw_distance_matrix(normalized_data)
    
    # 4. 执行 EPS 扫描或固定 EPS 聚类
    if SCAN_OPT_EPS:
        # 执行 EPS 扫描
        optimal_eps, eps_results = scan_eps(
            dist_matrix,
            CLUSTER_CONFIG['dbscan'],
            EPS_START,
            EPS_END,
            EPS_STEP
        )
        
        # 使用最优 EPS 执行最终聚类
        if optimal_eps:
            print(f"\n【最终聚类】")
            print(f"使用最优 eps: {optimal_eps:.2f}")
            labels = run_dbscan(dist_matrix, optimal_eps, CLUSTER_CONFIG['dbscan']['min_pts'])
            
            # 评估最终聚类结果
            metrics = evaluate_clustering(labels, dist_matrix)
            
            # 保存最终结果
            final_result = {
                'optimal_eps': optimal_eps,
                'metrics': metrics,
                'appliance_name': APPLIANCE_NAME,
                'timestamp': datetime.now().isoformat()
            }
            save_path = os.path.join(SAVE_DIR, 'optimal_clustering_result.json')
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            print(f"最终聚类结果已保存到: {save_path}")
    else:
        # 使用固定 EPS 聚类
        print("\n【固定 EPS 聚类】")
        eps = CLUSTER_CONFIG['dbscan']['eps']
        min_pts = CLUSTER_CONFIG['dbscan']['min_pts']
        labels = run_dbscan(dist_matrix, eps, min_pts)
        
        # 评估聚类结果
        metrics = evaluate_clustering(labels, dist_matrix)
        
        # 保存结果
        result = {
            'eps': eps,
            'metrics': metrics,
            'appliance_name': APPLIANCE_NAME,
            'timestamp': datetime.now().isoformat()
        }
        save_path = os.path.join(SAVE_DIR, 'clustering_result.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"聚类结果已保存到: {save_path}")
        
        # 调用聚类结果分析模块进行可视化
        print("\n【聚类结果可视化】")
        print("开始生成聚类结果可视化...")
        
        # 准备数据列表用于可视化
        data_list = []
        for i in range(min(len(labels), len(data_original))):
            data_list.append(data_original[i])
        
        # 保存聚类分析结果
        cluster_result_pic_save(data_list, seq_len, labels, save_dir=SAVE_DIR, col_index=K_INDEX)
        
        # 计算聚类评估指标并保存
        sil_score, db_score, ch_score = cluster_result_quantification(
            labels, dist_matrix, data_original, np.array(normalized_data), SAVE_DIR, 
            col_index=K_INDEX, visualize_noise=0
        )
        print("聚类结果可视化完成！")
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
