import json
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# ===============CONFIG=======================
BASE_DIR = r'./cluster_data/washing_machine_manually/'

# ================DeSTEC CONFIG=======================
CLUSTER_RESULT_FILE = f'cluster_data/detsec_clust_assignment.npy'
DATA_MAPPING_LIST = f'cluster_data/data_mapping_list.json'
ORIGINAL_DATA_FILE = f'cluster_data/data.npy'
SEQ_LEN_FILE = f'cluster_data/seq_length.npy'

# =================TIME GAP UNIT==================
HOURS = 3600
DAYS = 24 * HOURS
MONTHS = 30 * DAYS


def visualize_dict_data_layered(data_dict, title="Layered Visualization",
                                bar_width=0.8, x_axis=None, max_labels=5):
    """
    根据传入的字典做分层可视化，对每个key的value（np数组）以柱状图可视化

    :param data_dict: 包含数据的字典，key为子图标题，value为np数组
    :param title: 总标题
    :param bar_width: 柱状图宽度
    :param x_axis: x轴数据，如果非空则作为所有子图的x轴
    :param max_labels: 最大显示标签数量，用于控制x轴标签密度
    :return: matplotlib figure对象
    """

    # 获取字典的键值对数量
    n_items = len(data_dict)

    if n_items == 0:
        print("字典为空，无法可视化")
        return None

    # 计算子图布局
    subplot_rows = (n_items + 2 - 1) // 2

    # 动态计算图形大小
    figsize = (12, subplot_rows * 4)  # 每行4个单位高度

    # 创建图形和子图
    fig, axes = plt.subplots(subplot_rows, 2, figsize=figsize)

    # 处理只有一个子图的情况
    if n_items == 1:
        axes = [axes]
    elif subplot_rows == 1 and n_items < 2:
        axes = [axes] if isinstance(axes, plt.Axes) else axes.flatten()
    else:
        axes = axes.flatten() if n_items > 1 else [axes]

    # 生成颜色映射，为每个子图分配不同颜色
    colors = plt.cm.tab10(np.linspace(0, 1, n_items)) if n_items <= 10 else plt.cm.hsv(
        np.linspace(0, 1, n_items))

    # 为每个键值对创建柱状图
    for idx, (key, value) in enumerate(data_dict.items()):
        if isinstance(value, np.ndarray):
            ax = axes[idx]

            # 检查x_axis参数
            if x_axis is not None and isinstance(x_axis, np.ndarray):
                if len(x_axis) == len(value):
                    x_pos = x_axis
                else:
                    print(
                        f"Warning: x_axis length ({len(x_axis)}) does not match value length ({len(value)}) for key '{key}', using default range")
                    x_pos = np.arange(len(value))
            else:
                x_pos = np.arange(len(value))

            # 绘制柱状图，使用不同颜色
            ax.bar(x_pos, value, width=bar_width, color=colors[idx])

            # 设置标题和标签
            ax.set_title(f'Cluster_{key}')
            ax.set_xlabel('Index')
            ax.set_ylabel('Power')

            # 控制x轴标签密度 - 固定显示等距离标签
            n_ticks = len(x_pos)
            if max_labels > 0 and n_ticks > max_labels:
                # 计算等距离的索引位置
                indices = np.linspace(0, n_ticks - 1, max_labels, dtype=int)
                selected_ticks = [x_pos[i] for i in indices]
                ax.set_xticks(indices)
                ax.set_xticklabels(selected_ticks, rotation=45, ha='right')
            else:
                # 如果标签数量不超过最大限制，显示所有标签
                ax.set_xticks(range(len(x_pos)))
                if x_axis is not None:
                    ax.set_xticklabels(x_pos, rotation=45, ha='right')

            # 添加网格
            ax.grid(True, alpha=0.3)
        else:
            print(f"Warning: Value for key '{key}' is not a numpy array")

    # 隐藏多余的子图
    for idx in range(n_items, len(axes)):
        axes[idx].set_visible(False)

    # 设置总标题
    fig.suptitle(title, fontsize=16)

    # 调整子图布局
    plt.tight_layout()
    plt.show()

    return fig


def visualize_cluster_by_time_gap(data_mapping, cluster_result, time_gap=DAYS):
    """
    按时间间隔统计聚类结果并以直方图形式可视化
    :param data_mapping: 数据映射列表，包含时间戳信息
    :param cluster_result: 聚类结果数组
    :param time_gap: 时间间隔，默认为1天(3600*24)
    :return: matplotlib figure对象
    """
    # 获取时间范围
    total_start_time = data_mapping[0]['start_timestamp']
    total_end_time = data_mapping[len(cluster_result) - 1]['end_timestamp']

    # 计算时间区间
    time_bins = np.arange(total_start_time, total_end_time + time_gap, time_gap)
    n_bins = len(time_bins) - 1

    # 将每个bin的开始时间转换为YY/MM/DD格式的字符串数组
    bin_start_times = np.array([
        datetime.datetime.fromtimestamp(total_start_time + i * time_gap).strftime('%y/%m/%d')
        for i in range(n_bins)
    ])

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

        # 计算该段数据跨越的时间区间
        data_bin = int((start_time - total_start_time) / time_gap)
        data_bin = max(0, min(data_bin, n_bins - 1))
        duration = end_time - start_time
        cluster_time_stats[cluster_id][data_bin] += duration

    fig = visualize_dict_data_layered(cluster_time_stats, title="Cluster Time Statistics",
                                      x_axis=bin_start_times)

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


def cluster_result_save(data_array, seq_length, cluster_result, save_dir):
    """
    保存结果
    :return:
    """
    for i in range(len(data_array)):
        data = data_array[i][:seq_length[i]]
        cluster_id = cluster_result[i]
        plt.figure(figsize=(10, 6))
        plt.plot(data)
        plt.title(f"Cluster {cluster_id} - Item {i + 1}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        dir = save_dir + f'/cluster_{cluster_id}/'
        os.makedirs(dir, exist_ok=True)
        plt.savefig(dir + f'item_{i + 1}.png')
        plt.close()


def cluster_result_quantification(cluster_labels, dist_matrix, ts_data, save_dir=None):
    """
    对聚类结果进行量化，使用三种指标：
    - 轮廓系数（Silhouette Coefficient）：最直观的 “簇内紧凑 + 簇间分离” 度量
    - DB 指数（Davies-Bouldin Index）：基于 “簇中心距离” 的聚类纯度度量
    - CH 指数（Calinski-Harabasz Index）：基于 “方差比” 的分离度度量

    :return:
    """
    # ===================== 5. 预处理评估数据：过滤噪声点（关键步骤！） =====================
    # 筛选出非噪声的样本索引和对应标签
    valid_idx = cluster_labels != -1
    valid_dist_matrix = dist_matrix[valid_idx][:, valid_idx]  # 过滤后的DTW距离矩阵
    valid_labels = cluster_labels[valid_idx]

    # 正确处理ts_data的索引，将其转换为numpy数组以便正确索引
    ts_data_array = np.array(ts_data)
    valid_ts_data = ts_data_array[valid_idx]  # 过滤后的时序数据

    if len(cluster_labels) != len(ts_data):
        raise ValueError(f"标签数量({len(cluster_labels)})与时序数据数量({len(ts_data)})不匹配")

    # 跳过评估的异常情况：聚类结果只有1个簇 / 无有效样本
    n_clusters = len(np.unique(valid_labels))
    if n_clusters < 2:
        print("⚠️ 聚类结果仅生成1个有效簇，无法计算聚类评估指标！")
    else:
        # ===================== 6. 定量评估：三大核心指标计算（完整） =====================
        ## 6.1 轮廓系数 (Silhouette Coefficient) - 核心评估指标
        sil_score = silhouette_score(valid_dist_matrix, valid_labels, metric='precomputed')

        ## 6.2 DB指数 (Davies-Bouldin Index)
        db_score = davies_bouldin_score(valid_ts_data, valid_labels)

        ## 6.3 CH指数 (Calinski-Harabasz Index)
        ch_score = calinski_harabasz_score(valid_ts_data, valid_labels)

        # ===================== 7. 结果输出 + 指标解读 =====================
        print("=" * 60)
        print("时序数据DBSCAN-DTW聚类 定量评估结果")
        print("=" * 60)
        print(f"有效聚类样本数: {len(valid_labels)} | 噪声点数: {len(cluster_labels) - len(valid_labels)}")
        print(f"聚类簇数量: {len(np.unique(valid_labels))}")
        print("-" * 60)
        print(f"轮廓系数 (Silhouette) ：{sil_score:.4f} → 越接近1越好，>0.5为优秀")
        print(f"DB指数 (Davies-Bouldin)：{db_score:.4f} → 越接近0越好，<1.5为优秀")
        print(f"CH指数 (Calinski-Harabasz)：{ch_score:.2f} → 数值越大越好，无上限")
        print("=" * 60)

        from tslearn.barycenters import dtw_barycenter_averaging  # 时序专属：DTW重心计算

        # ========== 簇中心轮廓可视化核心代码（每个簇一个子图） ==========
        cluster_colors = plt.cm.tab10(np.arange(n_clusters))
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
        # 根据簇数量动态设置图形尺寸
        figsize_height = max(8, n_clusters * 2)  # 每个子图约2个单位高度，最少8个单位
        fig, axes = plt.subplots(n_clusters, 1, figsize=(12, figsize_height))
        fig.suptitle('时序聚类-各簇中心轮廓分布图 (DTW重心)', fontsize=14)

        # 处理只有一个子图的情况
        if n_clusters == 1:
            axes = [axes]
        elif n_clusters > 1:
            axes = axes.flatten()

        # 遍历每个簇，绘制中心轮廓
        for i, cluster_id in enumerate(np.unique(valid_labels)):  # 使用实际存在的簇ID
            # 筛选当前簇的所有时序样本
            cluster_seq = valid_ts_data[valid_labels == cluster_id]
            if len(cluster_seq) > 0:  # 确保簇中有数据
                # 计算当前簇的【DTW重心序列】- 时序聚类的最优中心，不是简单均值！
                cluster_center = dtw_barycenter_averaging(cluster_seq)
                # 绘制簇中心轮廓（加粗高亮，核心特征）
                axes[i].plot(cluster_center, color=cluster_colors[cluster_id % 10],
                             linewidth=2, label=f'簇 {cluster_id} 中心轮廓 (样本数:{len(cluster_seq)})')

                axes[i].set_title(f'簇 {cluster_id} 中心轮廓 (样本数: {len(cluster_seq)})', fontsize=12)
                axes[i].set_xlabel('时间步 / 序列长度', fontsize=10)
                axes[i].set_ylabel('时序数值', fontsize=10)
                axes[i].legend(fontsize=9)
                axes[i].grid(alpha=0.3)
            else:
                axes[i].text(0.5, 0.5, f'簇 {cluster_id} (样本数: 0)',
                             horizontalalignment='center', verticalalignment='center',
                             transform=axes[i].transAxes, fontsize=12)
                axes[i].set_title(f'簇 {cluster_id} - 无数据', fontsize=12)

        plt.tight_layout()
        plt.savefig(save_dir + 'cluster_center.png')
        plt.show()

        from sklearn.manifold import TSNE

        # ========== tSNE降维可视化核心代码（时序最优参数，无需调参） ==========
        # tsne核心参数：perplexity是关键，时序数据设为 样本数的1/10 即可，默认30足够用
        tsne = TSNE(
            n_components=2,  # 降维到2维，适合可视化
            perplexity=30,  # 核心参数，时序数据最优值：10~50，默认30
            random_state=42,  # 固定随机种子，结果可复现
            n_jobs=-1,  # 多核加速，计算更快
            init='pca'  # 初始化方式，避免局部最优
        )
        # 对高维时序数据降维 → 得到2维坐标
        tsne_2d = tsne.fit_transform(ts_data_array)

        # ========== 绘制tSNE聚类散点图 ==========
        plt.figure(figsize=(10, 8))
        # 绘制有效聚类样本
        for cluster_id in range(n_clusters):
            idx = (cluster_labels == cluster_id) & (cluster_labels != -1)
            plt.scatter(tsne_2d[idx, 0], tsne_2d[idx, 1], c=[cluster_colors[cluster_id]],
                        label=f'簇 {cluster_id + 1}', s=60, alpha=0.8)
        # 绘制噪声点
        noise_idx = cluster_labels == -1
        plt.scatter(tsne_2d[noise_idx, 0], tsne_2d[noise_idx, 1], c='black', marker='x', label='噪声点', s=80,
                    alpha=0.8)
        plt.title('时序聚类-tSNE降维分布图 (含噪声点)', fontsize=14)
        plt.legend()
        plt.savefig(save_dir + 'tsne.png')
        plt.show()
        return sil_score, db_score, ch_score


if __name__ == '__main__':
    # data_info_list, cluster_dict = read_detsec_result()
    # cluster_result_analyze(data_info_list, cluster_dict)
    with open(BASE_DIR + 'data_mapping_list.json', 'r', encoding='utf-8') as file:
        mapping_list = json.load(file)

    data = np.load(BASE_DIR + 'data.npy')
    cluster_result = np.load(BASE_DIR + 'dbscan_result_8_2.npy')
    seq_len = np.load(BASE_DIR + 'seq_length.npy')
    fig = visualize_cluster_by_time_gap(mapping_list, cluster_result)

    if fig is not None:
        # 显示图表
        plt.show()

        # 保存图表
        save_path = BASE_DIR + 'cluster_time_axis_visualization.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
