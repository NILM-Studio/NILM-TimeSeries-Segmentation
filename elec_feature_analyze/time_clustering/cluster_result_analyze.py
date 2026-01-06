import json
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
                    print(f"Warning: x_axis length ({len(x_axis)}) does not match value length ({len(value)}) for key '{key}', using default range")
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


def mapping_result_to_time_axis(data_np, seq_len, data_mapping, cluster_result):
    """
    mapping cluster result to time axis, save and visualize the output
    input the cluster result and output the cluster distribution in the time axis
    :return:
    """
    total_start_time = data_mapping[0]['start_timestamp']
    total_end_time = data_mapping[len(data_mapping) - 1]['end_timestamp']

    # 获取字典，key是cluster_id，value是所有该cluster的数据下标索引
    print("根据cluster分类数据中.....")
    unique_values, indices = np.unique(cluster_result, return_inverse=True)
    value_dict = {}

    for i, value in enumerate(unique_values):
        # 找到所有等于当前值的原始索引
        original_indices = np.where(indices == i)[0].tolist()
        value_dict[value] = original_indices

    # 创建一个时间轴，从total_start_time到total_end_time
    time_axis = np.arange(total_start_time, total_end_time + 1)
    # 为每个簇创建时间序列数据
    cluster_time_series = {}
    for cluster_id, index_list in value_dict.items():
        # 为当前簇创建时间序列，初始化为NaN（表示无数据）
        cluster_data = np.full_like(time_axis, np.nan, dtype=float)

        for i in index_list:
            start_time = data_mapping[i]['start_timestamp']
            end_time = data_mapping[i]['end_timestamp']
            data = data_np[i][:seq_len[i]].squeeze(-1)

            # 找到在时间轴上的对应位置
            start_idx = np.where(time_axis == start_time)[0]
            if len(start_idx) > 0:
                start_idx = start_idx[0]
                # 确保不超出时间轴范围
                end_idx = min(start_idx + len(data), len(time_axis))
                actual_len = end_idx - start_idx

                if actual_len > 0:
                    cluster_data[start_idx:end_idx] = data[:actual_len]

        cluster_time_series[cluster_id] = {
            'time_axis': time_axis,
            'data': cluster_data
        }

    # 计算子图布局
    n_clusters = len(cluster_time_series)
    if n_clusters == 0:
        print("没有聚类结果可显示")
        return

    # 计算合适的行列数
    n_cols = 2  # 每行显示2个簇
    n_rows = (n_clusters + 1) // 2  # 计算需要的行数

    # 创建高分辨率的子图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows), dpi=150)

    # 如果只有一个簇，axes不会是数组
    if n_clusters == 1:
        axes = [axes]
    elif n_rows == 1 and n_clusters < 2:
        axes = [axes] if isinstance(axes, plt.Axes) else axes
    else:
        axes = axes.flatten() if n_clusters > 1 else [axes]

    # 生成颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters)) if n_clusters <= 10 else plt.cm.hsv(
        np.linspace(0, 1, n_clusters))

    # 为每个簇绘制子图
    for idx, (cluster_id, series_data) in enumerate(cluster_time_series.items()):
        ax = axes[idx]

        # 使用特定颜色绘制时间序列数据
        ax.plot(series_data['time_axis'], series_data['data'],
                label=f'Cluster {cluster_id}', alpha=0.8, linewidth=1, color=colors[idx])

        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(f'Cluster {cluster_id} on Time Axis', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()

    # 隐藏多余的子图
    for idx in range(len(cluster_time_series), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()
    return plt


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
