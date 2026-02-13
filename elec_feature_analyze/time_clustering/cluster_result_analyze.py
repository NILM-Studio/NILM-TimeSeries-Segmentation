import json
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from tslearn.barycenters import dtw_barycenter_averaging
from sklearn.manifold import TSNE

# ================DeSTEC CONFIG=======================
CLUSTER_RESULT_FILE = f'cluster_data/detsec_clust_assignment.npy'
DATA_MAPPING_LIST = f'cluster_data/data_mapping_list.json'
ORIGINAL_DATA_FILE = f'cluster_data/data.npy'
SEQ_LEN_FILE = f'cluster_data/seq_length.npy'


def visualize_dict_data_layered(data_dict, title="Layered Visualization",
                                bar_width=0.8, x_axis=None, max_labels=5):
    """
    根据传入的字典做分层可视化，对每个key的value（np数组）以柱状图可视化

    :param data_dict: 包含数据的字典，key为子图标题，value为np数组
    :param title: 总标题
    :param bar_width: 柱状图宽度
    :param x_axis: x轴数据（必需），支持datetime格式的字符串列表
    :param max_labels: 最大显示标签数量，用于控制x轴标签密度
    :return: matplotlib figure对象
    """

    if x_axis is None:
        raise ValueError("x_axis参数不能为None，请提供x轴数据")

    # 获取字典的键值对数量
    n_items = len(data_dict)

    if n_items == 0:
        print("字典为空，无法可视化")
        return None

    # 计算合适的行列数，让子图横向铺开
    if n_items <= 2:
        n_cols = n_items
        n_rows = 1
    elif n_items <= 6:
        n_cols = 2
        n_rows = (n_items + n_cols - 1) // n_cols
    elif n_items <= 12:
        n_cols = 3
        n_rows = (n_items + n_cols - 1) // n_cols
    elif n_items <= 20:
        n_cols = 4
        n_rows = (n_items + n_cols - 1) // n_cols
    else:
        n_cols = 5
        n_rows = (n_items + n_cols - 1) // n_cols

    # 动态计算图形大小，增加宽度而不是高度
    figsize = (n_cols * 5, n_rows * 4)  # 每列5个单位宽度，每行4个单位高度

    # 创建图形和子图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=150)  # 设置较高的dpi以提高分辨率

    # 处理只有一个子图的情况
    if n_items == 1:
        axes = [[axes]] if not hasattr(axes, '__iter__') else axes
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes

    # 生成颜色映射，为每个子图分配不同颜色
    colors = plt.cm.tab10(np.linspace(0, 1, n_items)) if n_items <= 10 else plt.cm.hsv(
        np.linspace(0, 1, n_items))

    # 为每个键值对创建柱状图
    for idx, (key, value) in enumerate(data_dict.items()):
        if isinstance(value, np.ndarray):
            ax = axes[idx]

            if len(x_axis) > len(value):
                value = np.pad(value, (0, len(x_axis) - len(value)), 'constant', constant_values=0)
            elif len(x_axis) < len(value):
                raise ValueError(f"x_axis长度 ({len(x_axis)}) 小于value长度 ({len(value)})，无法处理")

            x_pos = np.arange(len(value))

            ax.bar(x_pos, value, width=bar_width, color=colors[idx])

            ax.set_title(f'Cluster_{key}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Power')

            n_ticks = len(x_pos)
            if max_labels > 0 and n_ticks > max_labels:
                indices = np.linspace(0, n_ticks - 1, max_labels, dtype=int)
                ax.set_xticks(indices)
                ax.set_xticklabels([x_axis[i] for i in indices], rotation=45, ha='right')
            else:
                ax.set_xticks(range(len(x_pos)))
                ax.set_xticklabels(x_axis, rotation=45, ha='right')

            ax.grid(True, alpha=0.3)
        else:
            print(f"Warning: Value for key '{key}' is not a numpy array")

    # 隐藏多余的子图（当子图数量不是n_rows * n_cols的整数倍时）
    if hasattr(axes, '__iter__') and len(axes) > n_items:
        for idx in range(n_items, len(axes)):
            if hasattr(axes[idx], 'set_visible'):
                axes[idx].set_visible(False)

    # 设置总标题
    fig.suptitle(title, fontsize=16)

    # 调整子图布局
    plt.tight_layout()
    plt.show()

    return fig


def visualize_cluster_by_time_gap(data_mapping, cluster_result, time_gap_type='days', time_gap_value=1, 
                                  max_duration=3600 * 24, save_json_path='./'):
    """
    按时间间隔统计聚类结果并以直方图形式可视化
    :param data_mapping: 数据映射列表，包含时间戳信息
    :param cluster_result: 聚类结果数组
    :param time_gap_type: 时间间隔类型，可选 'days' 或 'months'
    :param time_gap_value: 时间间隔值，如 1 表示1天或1个月
    :param max_duration:最大时间段跨度，用于判断是否有出问题的段，有些数据段处理错误会算出来一个段长达几百个小时
    :param save_json_path: JSON文件保存路径，如果提供则将统计数据保存为JSON格式
    :return: matplotlib figure对象
    """
    
    # 获取时间范围
    total_start_time = data_mapping[0]['start_timestamp']
    total_end_time = data_mapping[len(cluster_result) - 1]['end_timestamp']
    
    # 转换为datetime对象
    start_datetime = datetime.datetime.fromtimestamp(total_start_time)
    end_datetime = datetime.datetime.fromtimestamp(total_end_time)
    
    # 生成时间bins（包含最后一个bin的结束时间，用于统计）
    bin_start_datetimes = []
    current_datetime = start_datetime
    
    if time_gap_type == 'days':
        # 按天生成bins
        while current_datetime <= end_datetime:
            bin_start_datetimes.append(current_datetime)
            current_datetime += datetime.timedelta(days=time_gap_value)
    elif time_gap_type == 'months':
        # 按月生成bins
        while current_datetime <= end_datetime:
            bin_start_datetimes.append(current_datetime)
            # 处理月份递增
            if current_datetime.month == 12:
                current_datetime = current_datetime.replace(year=current_datetime.year + 1, month=1)
            else:
                current_datetime = current_datetime.replace(month=current_datetime.month + time_gap_value)
    else:
        raise ValueError("time_gap_type must be 'days' or 'months'")
    
    n_bins = len(bin_start_datetimes) - 1
    
    # 生成每个time_gap的起始时间和结束时间
    time_gap_start_datetimes = bin_start_datetimes[:n_bins]
    time_gap_end_datetimes = bin_start_datetimes[1:]
    
    # 转换为时间戳格式
    time_gap_start_timestamps = np.array([dt.timestamp() for dt in time_gap_start_datetimes])
    time_gap_end_timestamps = np.array([dt.timestamp() for dt in time_gap_end_datetimes])
    
    # 转换为字符串格式
    time_gap_start_datetimes_str = np.array([dt.strftime('%y/%m/%d') for dt in time_gap_start_datetimes])
    time_gap_end_datetimes_str = np.array([dt.strftime('%y/%m/%d') for dt in time_gap_end_datetimes])

    # 获取唯一聚类ID
    unique_clusters = np.unique(cluster_result)
    n_clusters = len(unique_clusters)

    # 初始化统计字典
    cluster_time_stats = {}
    for cluster_id in unique_clusters:
        cluster_time_stats[cluster_id] = np.zeros(n_bins)

    # 统计每个时间段内各聚类的总时长
    for i in range(len(cluster_result)):
        mapping_info = data_mapping[i]
        start_time = mapping_info['start_timestamp']
        end_time = mapping_info['end_timestamp']
        cluster_id = cluster_result[i]
        duration = end_time - start_time

        # 错误检测：如果duration超过24小时，打印对应的data_mapping信息并跳过
        if duration > max_duration:
            print(f"警告: 检测到超长duration ({duration}s) 对应的data_mapping信息:")
            print(f"  索引: {i}")
            print(f"  cluster_id: {cluster_id}")
            print(f"  start_timestamp: {start_time} ({datetime.datetime.fromtimestamp(start_time)})")
            print(f"  end_timestamp: {end_time} ({datetime.datetime.fromtimestamp(end_time)})")
            print(f"  duration: {duration}s ({duration / 3600:.2f}小时)")
            print("-" * 50)
            continue  # 跳过当前记录

        # 计算该段数据对应的时间bin
        start_datetime = datetime.datetime.fromtimestamp(start_time)
        data_bin = -1
        
        for j in range(n_bins):
            bin_start = bin_start_datetimes[j]
            bin_end = bin_start_datetimes[j + 1]
            if bin_start <= start_datetime < bin_end:
                data_bin = j
                break
        
        if data_bin >= 0 and data_bin < n_bins:
            cluster_time_stats[cluster_id][data_bin] += duration

    # 如果提供了保存路径，则将数据保存为JSON格式
    if save_json_path:
        json_data = {}
        for cluster_id in unique_clusters:
            # 为每个time_gap保存起始时间、结束时间和对应时长，使用List存储
            time_data = []
            for i in range(n_bins):
                time_gap_info = {
                    'start_timestamp': time_gap_start_timestamps[i],
                    'start_datetime': time_gap_start_datetimes_str[i],
                    'end_timestamp': time_gap_end_timestamps[i],
                    'end_datetime': time_gap_end_datetimes_str[i],
                    'interval_total_duration': cluster_time_stats[cluster_id][i]
                }
                time_data.append(time_gap_info)
            json_data[str(cluster_id)] = time_data

        # 保存到JSON文件
        if not os.path.exists(save_json_path):
            os.makedirs(save_json_path)
        
        with open(os.path.join(save_json_path, 'time_gap_data.json'), 'w', encoding='utf-8') as f:
            import json
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        print(f"统计数据已保存到: {save_json_path}")

    # 可视化
    fig = visualize_dict_data_layered(cluster_time_stats, title="Cluster Time Statistics",
                                      x_axis=time_gap_start_datetimes_str)

    return fig


def read_detsec_result():
    """
    读取DeTSEC的运行结果，并且进行结果映射
    :return:
    """
    cluster_result = np.load(CLUSTER_RESULT_FILE)
    data_len = np.load(SEQ_LEN_FILE)
    data = np.load(ORIGINAL_DATA_FILE)
    with open(DATA_MAPPING_LIST, 'r', encoding='utf-8') as file:
        data_info_list = json.load(file)

    print(cluster_result)
    cluster_dict = {}
    for i, data_info in enumerate(data_info_list):
        # add cluster res and original data
        data_info['cluster_id'] = cluster_result[i]
        data_info['data'] = pd.DataFrame(data[i][:data_len[i]])
        # create list if not exist (k,v)
        if cluster_result[i] not in cluster_dict:
            cluster_dict[cluster_result[i]] = []
        cluster_dict[cluster_result[i]].append(data_info)

    return data_info_list, cluster_dict


def cluster_result_analyze(data_info_list, cluster_dict):
    user_input = input("请输入命令:\n- show:遍历展示指定簇的数据 ")

    while user_input != 'e':
        if user_input == 'show':
            cluster_id = input("请输入簇ID: ")
            cluster_list = cluster_dict[int(cluster_id)]

            for i, item in enumerate(cluster_list):
                # 打印数据信息
                print(f"\n数据项 {i + 1}/{len(cluster_list)}")
                print(f"数据文件: {item.get('data_file', 'N/A')}")
                print(f"开始时间: {item.get('start_time', 'N/A')}")
                print(f"结束时间: {item.get('end_time', 'N/A')}")

                # 可视化数据
                plt.figure(figsize=(10, 6))
                plt.plot(item['data'])
                plt.title(f"Cluster {cluster_id} - Item {i + 1}")
                plt.xlabel("Time")
                plt.ylabel("Value")
                plt.show()

                # 等待用户输入继续
                input("按回车键查看下一个数据项...")
            print(f"簇 {cluster_id} 的所有数据项已展示完毕")
        else:
            print("无效的输入")


def cluster_result_save(data_array, seq_length, cluster_result, save_dir, threshold=200):
    """
    保存聚类结果，按照cluster保存所有的聚类结果，将所有时间序列片段可视化并且保存到其对应的
    cluster_id的文件夹下

    Parameters:
    data_array: array-like
        原始数据数组
    seq_length: array-like
        每个数据序列的长度
    cluster_result: array-like
        聚类结果数组
    save_dir: str
        保存根目录
    threshold: int
        每个聚类最多保存的图片数量
    """
    import shutil  # 导入shutil模块用于清空文件夹

    # 按照cluster对数据进行分组
    cluster_groups = {}
    for i in range(len(data_array)):
        cluster_id = cluster_result[i]
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(i)

    # 遍历每个cluster，保存对应的数据图像
    for cluster_id, indices in cluster_groups.items():
        # 如果该聚类的数量超过阈值，则只取前threshold个
        if len(indices) > threshold:
            indices = indices[:threshold]
            print(f"Cluster {cluster_id} 数据量过大，仅保存前 {threshold} 个")

        # 创建该cluster的目录
        dir = save_dir + f'/cluster_{cluster_id}/'

        # 如果目录存在，清空其中的内容
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir, exist_ok=True)

        # 收集该cluster的所有数据
        cluster_data = []
        for idx, data_idx in enumerate(indices):
            data = data_array[data_idx][:seq_length[data_idx]]
            cluster_data.append(data)

            # 保存该cluster的每个数据项
            plt.figure(figsize=(10, 6))
            plt.plot(data)
            plt.title(f"Cluster {cluster_id} - Item {idx + 1}")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.savefig(dir + f'item_{idx + 1}.png')
            plt.close()

        # 将整个cluster的数据保存为一个npy文件
        cluster_data_array = np.array(cluster_data, dtype=object)  # 使用object类型以支持不同长度的序列
        np.save(os.path.join(save_dir, f'Cluster_{cluster_id}.npy'), cluster_data_array)


def preprocess_cluster_data(
        cluster_labels: np.ndarray,
        dist_matrix: np.ndarray,
        org_data: np.ndarray,
        feature_matrix: np.ndarray
) -> tuple:
    """
    聚类评估数据预处理：过滤噪声点、校验数据合法性，区分org_data/feature_matrix的过滤逻辑
    :param cluster_labels: 原始聚类标签（含噪声点-1）
    :param dist_matrix: 预计算的距离矩阵（如DTW距离，.shape=(n_samples, n_samples)）
    :param org_data: 原始时序数据（.shape=(n_samples, seq_len)），用于簇中心计算
    :param feature_matrix: 特征矩阵（.shape=(n_samples, n_features)），用于评分和tSNE
    :return: 元组(valid_idx, valid_dist_matrix, valid_labels, valid_org_data, valid_feature_matrix, n_clusters)
             - valid_idx: 非噪声样本索引
             - valid_dist_matrix: 过滤后的距离矩阵
             - valid_labels: 过滤后的聚类标签
             - valid_org_data: 过滤后的原始时序数据（用于簇中心）
             - valid_feature_matrix: 过滤后的特征矩阵（用于评分）
             - n_clusters: 有效聚类簇数
    :raises ValueError: 数据维度不匹配时抛出
    """
    # 1. 基础维度校验
    n_samples = len(cluster_labels)
    if dist_matrix.shape != (n_samples, n_samples):
        raise ValueError(f"距离矩阵维度{dist_matrix.shape}与样本数{n_samples}不匹配（需为{n_samples}x{n_samples}）")
    if org_data.shape[0] != n_samples:
        raise ValueError(f"原始时序数据行数{org_data.shape[0]}与标签数{n_samples}不匹配")
    if feature_matrix.shape[0] != n_samples:
        raise ValueError(f"特征矩阵行数{feature_matrix.shape[0]}与标签数{n_samples}不匹配")

    # 2. 筛选非噪声样本
    valid_idx = cluster_labels != -1
    valid_dist_matrix = dist_matrix[valid_idx][:, valid_idx]  # 过滤距离矩阵（轮廓系数用）
    valid_labels = cluster_labels[valid_idx]  # 过滤标签
    valid_org_data = org_data[valid_idx]  # 过滤原始时序数据（簇中心用）
    valid_feature_matrix = feature_matrix[valid_idx]  # 过滤特征矩阵（评分用）

    # 3. 计算有效簇数
    n_clusters = len(np.unique(valid_labels))

    return valid_idx, valid_dist_matrix, valid_labels, valid_org_data, valid_feature_matrix, n_clusters


def calculate_cluster_metrics(
        valid_dist_matrix: np.ndarray,
        valid_labels: np.ndarray,
        valid_feature_matrix: np.ndarray,
        cluster_labels: np.ndarray
) -> tuple[float | None, float | None, float | None]:
    """
    计算聚类评估三大核心指标（严格使用feature_matrix），并打印量化结果
    :param valid_dist_matrix: 过滤噪声后的距离矩阵（轮廓系数用）
    :param valid_labels: 过滤噪声后的聚类标签
    :param valid_feature_matrix: 过滤噪声后的特征矩阵（DB/CH指数用）
    :param cluster_labels: 原始聚类标签（用于计算噪声点数）
    :return: sil_score, db_score, ch_score（无有效簇时返回None, None, None）
    """
    n_clusters = len(np.unique(valid_labels))

    # 异常情况：聚类数<2时无法计算指标
    if n_clusters < 2:
        print("⚠️ 聚类结果仅生成1个有效簇，无法计算聚类评估指标！")
        return None, None, None

    # 1. 轮廓系数（基于预计算距离矩阵，metric='precomputed'）
    sil_score = silhouette_score(valid_dist_matrix, valid_labels, metric='precomputed')
    # 2. DB指数（严格使用特征矩阵，而非原始时序数据）
    db_score = davies_bouldin_score(valid_feature_matrix, valid_labels)
    # 3. CH指数（严格使用特征矩阵，而非原始时序数据）
    ch_score = calinski_harabasz_score(valid_feature_matrix, valid_labels)

    # 打印量化结果（带解读）
    print("=" * 60)
    print("时序数据DBSCAN-DTW聚类 定量评估结果")
    print("=" * 60)
    print(f"有效聚类样本数: {len(valid_labels)} | 噪声点数: {len(cluster_labels) - len(valid_labels)}")
    print(f"聚类簇数量: {n_clusters}")
    print("-" * 60)
    print(f"轮廓系数 (Silhouette) ：{sil_score:.4f} → 越接近1越好，>0.5为优秀")
    print(f"DB指数 (Davies-Bouldin)：{db_score:.4f} → 越接近0越好，<1.5为优秀")
    print(f"CH指数 (Calinski-Harabasz)：{ch_score:.2f} → 数值越大越好，无上限")
    print("=" * 60)

    return sil_score, db_score, ch_score


def visualize_cluster_results(
        cluster_labels: np.ndarray,
        valid_labels: np.ndarray,
        valid_org_data: np.ndarray,
        feature_matrix: np.ndarray,
        save_dir: str | None = None,
        dist_method: str = 'dtw',
        sampling_threshold: int = 200  # 新增采样阈值参数
) -> None:
    """
    聚类结果可视化：
    - 簇中心轮廓图 → 用valid_org_data（原始时序数据）
    - tSNE降维图 → 用feature_matrix（特征矩阵）
    - 每个簇的前sampling_threshold个数据堆叠可视化 → 用valid_org_data（原始时序数据）
    :param dist_method: 距离计算方法
    :param cluster_labels: 原始聚类标签（含噪声点-1）
    :param valid_labels: 过滤噪声后的聚类标签
    :param valid_org_data: 过滤噪声后的原始时序数据（簇中心用）
    :param feature_matrix: 完整特征矩阵（含噪声点，tSNE用）
    :param save_dir: 可视化结果保存目录（None则不保存）
    :param sampling_threshold: 当簇中样本数量超过此阈值时，将进行采样以避免计算失败
    """
    n_clusters = len(np.unique(valid_labels))
    if n_clusters < 2:
        print("⚠️ 聚类簇数量<2，跳过可视化！")
        return

    print(f"开始可视化 {n_clusters} 个聚类的结果...")

    # 全局配置：中文显示 + 颜色映射
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    cluster_colors = plt.cm.tab10(np.arange(n_clusters))

    # ========== 1. 绘制簇中心轮廓图（严格使用原始时序数据org_data） ==========
    print("步骤1: 开始绘制簇中心轮廓图...")
    figsize_height = max(8, n_clusters * 2)  # 动态调整子图高度
    fig, axes = plt.subplots(n_clusters, 1, figsize=(12, figsize_height))
    fig.suptitle('时序聚类-各簇中心轮廓分布图 (DTW重心)', fontsize=14, fontweight='bold')

    # 兼容单簇/多簇的axes格式
    axes = [axes] if n_clusters == 1 else axes.flatten()

    # 遍历每个簇绘制DTW重心（仅用org_data）
    for i, cluster_id in enumerate(np.unique(valid_labels)):
        print(f"  正在处理簇 {cluster_id}...")
        cluster_mask = valid_labels == cluster_id
        cluster_seq = valid_org_data[cluster_mask]

        if len(cluster_seq) > 0:
            print(f"    簇 {cluster_id} 包含 {len(cluster_seq)} 个样本，正在计算{dist_method}重心...")

            # 如果簇的大小超过阈值，进行随机采样
            if len(cluster_seq) > sampling_threshold:
                print(f"    簇 {cluster_id} 样本数量({len(cluster_seq)})超过阈值({sampling_threshold})，进行随机采样...")
                np.random.seed(42)  # 设置随机种子确保结果可重现
                sampled_indices = np.random.choice(len(cluster_seq), size=sampling_threshold, replace=False)
                cluster_seq = cluster_seq[sampled_indices]
                print(f"    采样完成，使用 {len(cluster_seq)} 个样本计算重心")

            # 计算重心，根据距离方法选择算法
            if dist_method == 'dtw':
                # 计算DTW重心（时序聚类专属，非简单均值）
                cluster_center = dtw_barycenter_averaging(cluster_seq)
            else:  # 默认使用欧几里得距离
                # 计算欧几里得空间的重心（对时序数据直接求均值）
                min_len = min(len(s) for s in cluster_seq)
                center = np.mean([s[:min_len] for s in cluster_seq], axis=0)
                cluster_center = center
            print(f"    {dist_method}重心计算完成")
            # 绘制重心轮廓
            axes[i].plot(cluster_center, color=cluster_colors[cluster_id % 10],
                         linewidth=2.5,
                         label=f'簇 {cluster_id} (样本数:{len(valid_org_data[valid_labels == cluster_id])})')
            axes[i].set_title(f'簇 {cluster_id} 中心轮廓', fontsize=12)
            axes[i].set_xlabel('时间步 / 序列长度', fontsize=10)
            axes[i].set_ylabel('时序数值', fontsize=10)
            axes[i].legend(fontsize=9)
            axes[i].grid(alpha=0.3, linestyle='--')
        else:
            axes[i].text(0.5, 0.5, f'簇 {cluster_id} (无样本)',
                         horizontalalignment='center', verticalalignment='center',
                         transform=axes[i].transAxes, fontsize=12)
            axes[i].set_title(f'簇 {cluster_id} - 无数据', fontsize=12)

    # 保存簇中心图
    print("  正在保存簇中心图...")
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/cluster_center.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("簇中心轮廓图绘制完成!")

    # ========== 2. 绘制所有簇的前sampling_threshold个数据堆叠可视化图（整合到一张图中） ==========
    print("步骤2: 开始绘制所有簇的前sampling_threshold个数据堆叠图...")

    # 计算子图布局
    n_cols = min(3, n_clusters)  # 最多3列
    n_rows = (n_clusters + n_cols - 1) // n_cols  # 计算所需行数

    # 创建整体图形
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    fig.suptitle('各簇数据堆叠可视化', fontsize=16, fontweight='bold')

    # 处理单子图情况
    if n_clusters == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_clusters > 1 else [axes]

    for i, cluster_id in enumerate(np.unique(valid_labels)):
        print(f"  正在处理簇 {cluster_id} 的数据堆叠可视化...")
        cluster_mask = valid_labels == cluster_id
        cluster_seq = valid_org_data[cluster_mask]

        if len(cluster_seq) > 0:
            # 限制要可视化的数据量为前sampling_threshold个
            if len(cluster_seq) > sampling_threshold:
                cluster_seq_subset = cluster_seq[:sampling_threshold]
                print(f"    簇 {cluster_id} 包含 {len(cluster_seq)} 个样本，将可视化前 {sampling_threshold} 个样本")
            else:
                cluster_seq_subset = cluster_seq
                print(f"    簇 {cluster_id} 包含 {len(cluster_seq)} 个样本，全部可视化")

            # 在对应的子图中绘制这个簇的所有序列
            for j, series in enumerate(cluster_seq_subset):
                axes[i].plot(series, alpha=0.6, label=f'Series {j}' if j < 3 else "")  # 只给前几个序列加label避免图例过长

            axes[i].set_title(f'Cluster {cluster_id} (前{len(cluster_seq_subset)}个数据)', fontsize=12)
            axes[i].set_xlabel('时间步 / 序列长度', fontsize=10)
            axes[i].set_ylabel('时序数值', fontsize=10)
            axes[i].grid(alpha=0.3, linestyle='--')
        else:
            axes[i].text(0.5, 0.5, f'簇 {cluster_id} (无样本)',
                         horizontalalignment='center', verticalalignment='center',
                         transform=axes[i].transAxes, fontsize=12)
            axes[i].set_title(f'簇 {cluster_id} - 无数据', fontsize=12)

    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # 保存堆叠图
    print("  正在保存堆叠图...")
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/clusters_stacked.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("所有簇的堆叠可视化图绘制完成!")

    # ========== 3. 绘制tSNE降维散点图（严格使用特征矩阵feature_matrix） ==========
    print("步骤3: 开始绘制tSNE降维散点图...")
    print(f"  准备进行tSNE降维，数据形状: {feature_matrix.shape}")

    # TSNE参数：时序特征最优配置（固定随机种子保证可复现）
    tsne = TSNE(
        n_components=2, perplexity=min(30, len(feature_matrix) // 10),  # 自适应perplexity
        random_state=42, n_jobs=-1, init='pca'
    )
    print("  开始tSNE拟合变换...")
    tsne_2d = tsne.fit_transform(feature_matrix)  # 用完整特征矩阵（含噪声点）
    print(f"  tSNE变换完成，结果形状: {tsne_2d.shape}")

    # 绘制散点图
    plt.figure(figsize=(10, 8))
    print("  开始绘制散点图...")

    # 绘制有效聚类样本
    for cluster_id in np.unique(valid_labels):
        idx = (cluster_labels == cluster_id) & (cluster_labels != -1)
        plt.scatter(tsne_2d[idx, 0], tsne_2d[idx, 1],
                    c=[cluster_colors[cluster_id % 10]], label=f'簇 {cluster_id}',
                    s=70, alpha=0.8, edgecolors='white', linewidth=0.5)
    # 绘制噪声点
    noise_idx = cluster_labels == -1
    plt.scatter(tsne_2d[noise_idx, 0], tsne_2d[noise_idx, 1],
                c='black', marker='x', label='噪声点', s=90, alpha=0.8)

    plt.title('时序聚类-tSNE降维分布图 (特征矩阵)', fontsize=14, fontweight='bold')
    plt.xlabel('tSNE维度1', fontsize=11)
    plt.ylabel('tSNE维度2', fontsize=11)
    plt.legend(fontsize=10, loc='best')
    plt.grid(alpha=0.2, linestyle='--')

    # 保存tSNE图
    print("  正在保存tSNE图...")
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/tsne.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("tSNE降维散点图绘制完成!")
    print("可视化完成!")


def cluster_result_quantification(
        cluster_labels: np.ndarray,
        dist_matrix: np.ndarray,
        org_data: np.ndarray,
        feature_matrix: np.ndarray,
        save_dir: str | None = None,
        visualize: bool = True
) -> tuple[float | None, float | None, float | None]:
    """
    聚类结果量化主入口（整合预处理、指标计算、可视化），完全兼容原接口
    :param cluster_labels: 原始聚类标签（含噪声点-1）
    :param dist_matrix: 预计算的距离矩阵（如DTW距离）
    :param org_data: 原始时序数据（用于簇中心可视化）
    :param feature_matrix: 特征矩阵（用于评分计算和tSNE可视化）
    :param save_dir: 可视化结果保存目录（None则不保存）
    :return: sil_score, db_score, ch_score
    """
    # 1. 数据预处理
    valid_idx, valid_dist_matrix, valid_labels, valid_org_data, valid_feature_matrix, n_clusters = preprocess_cluster_data(
        cluster_labels=cluster_labels,
        dist_matrix=dist_matrix,
        org_data=org_data,
        feature_matrix=feature_matrix
    )

    # 2. 计算评估指标
    sil_score, db_score, ch_score = calculate_cluster_metrics(
        valid_dist_matrix=valid_dist_matrix,
        valid_labels=valid_labels,
        valid_feature_matrix=valid_feature_matrix,
        cluster_labels=cluster_labels
    )

    # 3. 可视化（仅当有有效簇时执行）
    if n_clusters >= 2 and visualize:
        visualize_cluster_results(
            cluster_labels=cluster_labels,
            valid_labels=valid_labels,
            valid_org_data=valid_org_data,
            feature_matrix=feature_matrix,
            save_dir=save_dir
        )

    return sil_score, db_score, ch_score


if __name__ == '__main__':
    # # data_info_list, cluster_dict = read_detsec_result()
    # # cluster_result_analyze(data_info_list, cluster_dict)
    TIME_UNITS = 'months'
    DATA_DIR = r'./cluster_data/washing_machine_freq/'
    ANALYZE_DIR = r'./cluster_data/dbscan_result/washing_machine_freq/0.1_20_low_freq_bilistm/'
    with open(DATA_DIR + 'data_mapping_list.json', 'r', encoding='utf-8') as file:
        mapping_list = json.load(file)

    data = np.load(r'./cluster_data/washing_machine_freq/' + 'data.npy')
    cluster_result = np.load(ANALYZE_DIR + 'cluster_result_0.1_20_low_freq_bilistm.npy')
    seq_len = np.load(DATA_DIR + 'seq_length.npy')

    # 打印各个数组的形状以便调试
    print(f"data shape: {data.shape}")
    print(f"cluster_result shape: {cluster_result.shape}")
    print(f"seq_len shape: {seq_len.shape}")
    print(f'mapping_list length: {len(mapping_list)}')

    if not (len(data) == len(cluster_result) == len(seq_len) == len(mapping_list)):
        raise ValueError(
            f"数据长度不匹配: data长度={len(data)}, cluster_result长度={len(cluster_result)}, seq_len长度={len(seq_len)}")
    MAX_LEN = data.shape[1] * 6
    fig = visualize_cluster_by_time_gap(mapping_list, cluster_result, TIME_UNITS, 1, MAX_LEN, ANALYZE_DIR)

    if fig is not None:
        # 显示图表
        plt.show()

        # 保存图表
        save_path = './cluster_data/dbscan_result/washing_machine_freq/0.1_20_low_freq_bilistm/' + \
                    f'cluster_time_axis_visualization_{TIME_UNITS}.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
