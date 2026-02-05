from runpy import run_path
import json, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def modified_z_score(x):
    """计算改进版Z-score（鲁棒，基于中位数和MAD）"""
    median = np.median(x)
    mad = np.median(np.abs(x - median))  # 绝对中位偏差
    if mad == 0:
        return np.zeros_like(x)
    m_z_score = 0.6745 * (x - median) / mad
    return m_z_score


def sliding_window_threshold(ts, window_size=10, n=3):
    """计算滑动窗口的低值阈值（均值-n*标准差）
    参数:
    - ts: 时间序列数据（pd.Series）
    - window_size: 滑动窗口大小
    - n: 标准差倍数（n=3 对应均值-3σ）
    返回:
    - lower_threshold: 每个位置的滑动窗口低值阈值
    """
    rolling_mean = ts.rolling(window=window_size, center=True).mean()  # 中心窗口更贴合
    rolling_std = ts.rolling(window=window_size, center=True).std()
    # 填充窗口边缘的NaN（用整体均值/标准差替代）
    rolling_mean = rolling_mean.fillna(ts.mean())
    rolling_std = rolling_std.fillna(ts.std())
    lower_threshold = rolling_mean - n * rolling_std
    return lower_threshold


def visualize_significantly_low_points(cluster_df_list, title=None, titles=None, show_only_abnormal=True, save_dir=None):
    """
    可视化多个聚类的显著低值检测结果，以子图形式展示
    
    参数:
    - cluster_dfs: 聚类DataFrame列表或单个DataFrame，每个DataFrame包含interval_total_duration和is_significantly_low列
    - title: 单个图表的标题（向后兼容）
    - titles: 子图标题列表，与cluster_dfs一一对应
    - show_only_abnormal: 是否仅显示异常值（零值+显著低值）的标注
    - save_dir: 保存图表的目录路径
    """

    
    # 检查输入
    if not cluster_df_list:
        print("没有输入数据需要可视化")
        return
    
    # 如果输入是单个DataFrame，转换为列表
    if isinstance(cluster_df_list, pd.DataFrame):
        cluster_df_list = [cluster_df_list]
        if title:
            titles = [title]
        elif titles:
            titles = [titles]
    
    # 计算子图布局
    n_clusters = len(cluster_df_list)
    n_cols = min(2, n_clusters)  # 最多2列
    n_rows = (n_clusters + n_cols - 1) // n_cols  # 向上取整计算行数
    
    # 创建图形和子图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 6 * n_rows))
    axes = np.array(axes).reshape(-1)  # 确保axes是一维数组，方便遍历
    
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 遍历每个聚类DataFrame绘制子图
    for i, (cluster_df, ax) in enumerate(zip(cluster_df_list, axes)):
        # 获取数据
        duration_list = cluster_df['interval_total_duration']
        is_significantly_low = cluster_df['is_significantly_low']
        is_zero = (duration_list == 0)
        
        # 定义颜色映射
        colors = []
        for j in range(len(duration_list)):
            if is_zero.iloc[j]:
                colors.append('red')  # 0值用红色表示
            elif is_significantly_low.iloc[j]:
                colors.append('orange')  # 显著低值用橙色表示
            else:
                colors.append('blue')  # 正常值用蓝色表示
        
        # 绘制柱状图
        x_pos = range(len(duration_list))
        bars = ax.bar(x_pos, duration_list, color=colors, alpha=0.7)
        
        # 添加标注
        for j, (x, y, is_z, is_low) in enumerate(zip(x_pos, duration_list, is_zero, is_significantly_low)):
            if is_z:
                ax.text(x, 0.1, '0', ha='center', va='bottom', color='red', fontweight='bold', fontsize=6)
                ax.scatter(x, 0.5, color='red', marker='*', s=40, zorder=5)
            elif is_low:
                label_y = y + (max(duration_list) * 0.01) if max(duration_list) != 0 else y + 0.1
                ax.text(x, label_y, f'{y:.1f}', ha='center', va='bottom',
                        color='darkorange', fontweight='bold', fontsize=6)
            elif not show_only_abnormal:
                label_y = y + (max(duration_list) * 0.01) if max(duration_list) != 0 else y + 0.1
                ax.text(x, label_y, f'{y:.1f}', ha='center', va='bottom',
                        color='black', fontsize=5)
        
        # 添加图例（仅在第一个子图显示）
        if i == 0:
            legend_elements = [
                Patch(facecolor='blue', label='正常值'),
                Patch(facecolor='orange', label='显著低值'),
                Patch(facecolor='red', label='零值（★标记）')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
        
        # 设置标题和轴标签
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=12)
        else:
            ax.set_title(f'Cluster {i}', fontsize=12)
        ax.set_xlabel('数据点索引', fontsize=10)
        ax.set_ylabel('持续时间', fontsize=10)
        
        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 打印统计信息
        print(f"\nCluster {i} 统计信息:")
        print(f"总数据点数: {len(duration_list)}")
        print(f"显著低值点数: {sum(is_significantly_low)}")
        print(f"零值点数: {sum(is_zero)}")
        print(f"正常值点数: {len(duration_list) - sum(is_significantly_low) - sum(is_zero)}")
    
    # 隐藏多余的子图（如果有的话）
    for i in range(n_clusters, len(axes)):
        axes[i].set_visible(False)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表（如果指定了保存目录）
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'few_shot_cluster_visualization.png'), dpi=300, bbox_inches='tight')
        print(f"\n图表已保存到: {os.path.join(save_dir, 'few_shot_cluster_visualization.png')}")
    
    # 显示图表
    plt.show()


def find_consecutive_segments(cluster_df: pd.DataFrame) -> dict:
    """
    找到连续的高值段和低值段

    参数:
    - cluster_df: 包含is_significantly_low列的DataFrame

    返回:
    - segments_info: 包含高值段和低值段的字典，格式：
      {
        'low_segments': 连续显著低值段信息列表,
        'high_segments': 连续高值段信息列表
      }
      每个段信息包含开始索引、结束索引、总duration、时间戳等
    """

    def process_segments(indices):
        """处理连续段的辅助函数"""
        segments = []
        if len(indices) > 0:
            # 分组连续的索引
            consecutive_groups = []
            current_group = [indices[0]]

            for i in range(1, len(indices)):
                if indices[i] == indices[i - 1] + 1:  # 如果是连续索引
                    current_group.append(indices[i])
                else:  # 如果不是连续的，保存当前组并开始新组
                    consecutive_groups.append(current_group)
                    current_group = [indices[i]]

            # 添加最后一组
            consecutive_groups.append(current_group)

            # 为每组创建段信息
            for group in consecutive_groups:
                start_idx = group[0]
                end_idx = group[-1]

                # 计算总duration
                total_duration = cluster_df.loc[group, 'interval_total_duration'].sum()

                # 获取起始和结束时间戳
                start_timestamp = cluster_df.loc[start_idx, 'start_timestamp'] if start_idx < len(cluster_df) else None
                end_timestamp = cluster_df.loc[end_idx, 'end_timestamp'] if end_idx < len(cluster_df) else None

                # 获取当前连续段所包含的所有group信息
                group_info_list = cluster_df.loc[group].to_dict('records')
                
                segment_info = {
                    'start_index': start_idx,
                    'end_index': end_idx,
                    'total_duration': total_duration,
                    'segment_length': len(group),  # 段长度（连续点的数量）
                    'start_timestamp': int(start_timestamp),
                    'end_timestamp': int(end_timestamp),
                    'group_info': group_info_list  # 当前连续段包含的所有group信息
                }
                segments.append(segment_info)
        return segments

    # 找到所有显著低值的索引
    low_indices = cluster_df[cluster_df['is_significantly_low']].index.tolist()
    # 找到所有高值的索引（通过取反）
    high_indices = cluster_df[~cluster_df['is_significantly_low']].index.tolist()

    # 处理低值段和高值段
    low_segments = process_segments(low_indices)
    high_segments = process_segments(high_indices)

    # 返回包含两种类型段的字典
    segments_info = {
        'low_segments': low_segments,
        'high_segments': high_segments
    }

    return segments_info


def get_dataset_df(consecutive_segments: dict, data_df, low_threshold, test_dict) -> dict:
    """
    根据连续段信息和阈值选择段，切割数据源生成训练集、验证集和测试集

    参数:
    - consecutive_segments: 包含高值段和低值段的字典，格式：
      {
        'low_segments': 连续显著低值段信息列表,
        'high_segments': 连续高值段信息列表
      }
    - data_df: 数据源DataFrame，包含start_timestamp和end_timestamp列
    - low_threshold: 训练集累积总时长阈值，用于选择低值段
    - test_dict: 测试集构建配置字典，格式：
      {
        'test_method': 'total'或'duration',
        'total': {
            'time_unit': 'days'或'months',
            'time_length': 时间长度
        },
        'duration': {
            'threshold': 阈值
        }
      }

    说明:
        - 低值段按total_duration从低到高排序，逐个遍历group累积选择直到超过low_threshold
        - 高值段按total_duration从高到低排序，逐个遍历group累积选择直到满足test_dict配置的条件
        - 测试集构建方式：通过test_dict配置综合构建
        - 验证集构建方式：混合采样策略，确保包含小样本分布，同时保持时间顺序
        - 根据选择的group的start_timestamp和end_timestamp从data_df中提取数据并拼接生成训练集、验证集和测试集

    返回:
    - result: 包含排序段和选择group信息的字典，格式：
      {
        'sorted_low_segments': 按total_duration从低到高排序的低值段,
        'sorted_high_segments': 按total_duration从高到低排序的高值段,
        'selected_low_groups': 累积总时长超过low_threshold的低值group列表,
        'selected_high_groups': 选择的高值group列表,
        'train_groups': 最终的训练集groups列表,
        'val_groups': 最终的验证集groups列表,
        'low_total_duration': 选择的低值group总时长,
        'high_total_duration': 选择的高值group总时长,
        'test_dict': 使用的测试集构建配置
      }
    - train_data_df: 训练集DataFrame，由选择的低值段数据拼接而成
    - val_data_df: 验证集DataFrame，由选择的低值段数据拼接而成
    - test_data_df: 测试集DataFrame，由选择的高值段数据拼接而成
    """
    print("\n===== 开始执行数据切分 =====")
    # 检查参数是否为None
    if data_df is None or low_threshold is None or test_dict is None:
        raise ValueError("data_df、low_threshold和test_dict存在空参数")

    # 对low_segments按total_duration从低到高排序
    sorted_low_segments = sorted(
        consecutive_segments.get('low_segments', []),
        key=lambda x: x.get('total_duration', 0)
    )

    # 对high_segments按total_duration从高到低排序
    sorted_high_segments = sorted(
        consecutive_segments.get('high_segments', []),
        key=lambda x: x.get('total_duration', 0),
        reverse=True
    )
    
    print(f"段排序完成：低值段 {len(sorted_low_segments)} 个，高值段 {len(sorted_high_segments)} 个")

    # 处理低值段阈值 - 按group累积
    selected_low_groups = []
    low_total = 0
    
    # 遍历所有sorted_low_segments中的group
    for segment in sorted_low_segments:
        group_info = segment.get('group_info', [])
        for group in group_info:
            if low_total <= low_threshold:
                selected_low_groups.append(group)
                low_total += group.get('interval_total_duration', 0)
            else:
                break
        if low_total > low_threshold:
            break
    
    print(f"低值段选择完成：共选择 {len(selected_low_groups)} 个group，总时长 {low_total:.2f}")

    # 处理高值段 - 按group累积
    selected_high_groups = []
    high_total = 0
    
    # 使用test_dict配置构建测试集
    test_method = test_dict.get('test_method')
    
    if test_method == 'total':
        # 按时间长度累积
        time_unit = test_dict.get('total', {}).get('time_unit')
        time_length = test_dict.get('total', {}).get('time_length')
        
        if time_unit not in ['days', 'months']:
            raise ValueError("time_unit参数值无效，可选值为'days'或'months'")
        
        if time_unit == 'days':
            # 转换为秒
            total_seconds = time_length * 24 * 3600
        else:  # months
            # 近似值：1个月 = 30天
            total_seconds = time_length * 30 * 24 * 3600
        
        cumulative_time = 0
        for segment in sorted_high_segments:
            group_info = segment.get('group_info', [])
            for group in group_info:
                start_ts = group.get('start_timestamp')
                end_ts = group.get('end_timestamp')
                if start_ts is not None and end_ts is not None:
                    group_duration = end_ts - start_ts
                    if cumulative_time + group_duration <= total_seconds:
                        selected_high_groups.append(group)
                        cumulative_time += group_duration
                        high_total += group.get('interval_total_duration', 0)
                    else:
                        break
            if cumulative_time > total_seconds:
                break
    elif test_method == 'duration':
        # 按阈值累积
        threshold = test_dict.get('duration', {}).get('threshold')
        if threshold is None:
            raise ValueError("使用duration方法时，必须在test_dict中提供threshold值")
        
        for segment in sorted_high_segments:
            group_info = segment.get('group_info', [])
            for group in group_info:
                if high_total <= threshold:
                    selected_high_groups.append(group)
                    high_total += group.get('interval_total_duration', 0)
                else:
                    break
            if high_total > threshold:
                break
    else:
        raise ValueError("test_method参数值无效，可选值为'total'或'duration'")
    
    print(f"高值段选择完成：共选择 {len(selected_high_groups)} 个group，总时长 {high_total:.2f}")

    # 按start_timestamp排序selected_low_groups
    selected_low_groups.sort(key=lambda x: x.get('start_timestamp', 0))
    
    # 按start_timestamp排序selected_high_groups
    selected_high_groups.sort(key=lambda x: x.get('start_timestamp', 0))

    # 从data_df中提取高值段数据（测试集）
    high_segments_data_list = []
    for group in selected_high_groups:
        start_ts = group.get('start_timestamp')
        end_ts = group.get('end_timestamp')
        if start_ts is not None and end_ts is not None:
            # 根据时间戳从data_df中获取数据
            segment_data = data_df[(data_df['timestamp'] >= start_ts) & (data_df['timestamp'] <= end_ts)]
            high_segments_data_list.append(segment_data)
    # 拼接所有切割后的数据
    if high_segments_data_list:
        test_data_df = pd.concat(high_segments_data_list, ignore_index=True)
    else:
        test_data_df = pd.DataFrame()

    # 划分验证集 - 混合采样策略
    def split_train_val(train_groups, val_ratio=0.2):
        """
        划分训练集和验证集，确保验证集包含小样本分布，同时保持时间顺序
        
        参数:
        - train_groups: 训练集groups列表
        - val_ratio: 验证集比例
        
        返回:
        - train_groups_final: 最终的训练集groups
        - val_groups_final: 最终的验证集groups
        """
        if not train_groups:
            return [], []
        
        # 按时间排序
        sorted_groups = sorted(train_groups, key=lambda x: x.get('start_timestamp', 0))
        
        # 计算验证集大小
        val_size = max(1, int(len(sorted_groups) * val_ratio))
        print(f"\n训练集验证集切分：总groups数 {len(sorted_groups)}，验证集比例 {val_ratio}，验证集大小 {val_size}")
        
        # 初步划分：取最后val_size个groups作为候选验证集
        candidate_val_groups = sorted_groups[-val_size:]
        candidate_train_groups = sorted_groups[:-val_size]
        
        # 检查候选验证集是否包含小样本分布（这里假设is_significantly_low=True表示小样本分布）
        has_small_sample = any(group.get('is_significantly_low', False) for group in candidate_val_groups)
        print(f"候选验证集是否包含小样本分布：{has_small_sample}")
        
        if has_small_sample:
            # 候选验证集包含小样本分布，直接使用
            print(f"使用候选验证集：训练集 {len(candidate_train_groups)} 个group，验证集 {len(candidate_val_groups)} 个group")
            return candidate_train_groups, candidate_val_groups
        else:
            # 候选验证集不包含小样本分布，从候选训练集中选择包含小样本分布的groups
            small_sample_groups = [g for g in candidate_train_groups if g.get('is_significantly_low', False)]
            
            if small_sample_groups:
                # 选择时间最晚的小样本分布group
                latest_small_sample = max(small_sample_groups, key=lambda x: x.get('start_timestamp', 0))
                
                # 从候选训练集中移除该group，添加到候选验证集
                candidate_train_groups = [g for g in candidate_train_groups if g != latest_small_sample]
                candidate_val_groups.append(latest_small_sample)
                
                # 对候选验证集重新按时间排序
                candidate_val_groups.sort(key=lambda x: x.get('start_timestamp', 0))
                print(f"从训练集迁移小样本到验证集：训练集 {len(candidate_train_groups)} 个group，验证集 {len(candidate_val_groups)} 个group")
            else:
                print(f"未找到小样本分布，使用默认划分：训练集 {len(candidate_train_groups)} 个group，验证集 {len(candidate_val_groups)} 个group")
            
            return candidate_train_groups, candidate_val_groups
    
    # 划分训练集和验证集
    train_groups, val_groups = split_train_val(selected_low_groups)
    
    # 从data_df中提取训练集数据
    train_segments_data_list = []
    for group in train_groups:
        start_ts = group.get('start_timestamp')
        end_ts = group.get('end_timestamp')
        if start_ts is not None and end_ts is not None:
            segment_data = data_df[(data_df['timestamp'] >= start_ts) & (data_df['timestamp'] <= end_ts)]
            train_segments_data_list.append(segment_data)
    if train_segments_data_list:
        final_train_data_df = pd.concat(train_segments_data_list, ignore_index=True)
    else:
        final_train_data_df = pd.DataFrame()
    
    # 从data_df中提取验证集数据
    val_segments_data_list = []
    for group in val_groups:
        start_ts = group.get('start_timestamp')
        end_ts = group.get('end_timestamp')
        if start_ts is not None and end_ts is not None:
            segment_data = data_df[(data_df['timestamp'] >= start_ts) & (data_df['timestamp'] <= end_ts)]
            val_segments_data_list.append(segment_data)
    if val_segments_data_list:
        val_data_df = pd.concat(val_segments_data_list, ignore_index=True)
    else:
        val_data_df = pd.DataFrame()
    
    print(f"\n数据提取完成：")
    print(f"训练集大小：{len(final_train_data_df)} 条记录")
    print(f"验证集大小：{len(val_data_df)} 条记录")
    print(f"测试集大小：{len(test_data_df)} 条记录")
    
    print("\n===== 数据切分完成 =====")
    
    # 初始化结果字典
    result = {
        'sorted_low_segments': sorted_low_segments,
        'sorted_high_segments': sorted_high_segments,
        'selected_low_groups': selected_low_groups,  # 选择的低值group
        'selected_high_groups': selected_high_groups,  # 选择的高值group
        'train_groups': train_groups,  # 最终的训练集groups
        'val_groups': val_groups,  # 最终的验证集groups
        'low_total_duration': low_total,
        'high_total_duration': high_total,
        'test_dict': test_dict
    }

    return result, final_train_data_df, val_data_df, test_data_df


def few_shot_and_bias_statistics(cluster_df: pd.DataFrame, basic_threshold: int = 3000):
    """
    基于统计方法检测显著低值（多指标融合判定）
    参数:
    - cluster_df: 包含interval_total_duration列的DataFrame
    - basic_threshold: 基础低值阈值（兜底用）
    返回:
    - cluster_df: 新增各类判定字段+最终is_significantly_low的DataFrame
    - consecutive_segments: 连续显著低值段信息列表，每个元素为包含开始时间、结束时间、总duration的dict
    """
    duration_list = cluster_df['interval_total_duration']
    mean = np.mean(duration_list)
    std = np.std(duration_list)

    # 指标1：均值-2σ阈值（适合平稳正态序列，原注释3σ与代码2σ统一）
    threshold_sigma = mean - 2 * std  # 正常低值下限
    cluster_df['is_low_3sigma'] = duration_list < min(threshold_sigma, basic_threshold)

    # 指标2：中位数-2×MAD阈值（鲁棒，抗极端高值干扰）
    median = duration_list.median()
    mad = np.median(np.abs(duration_list - median))
    threshold_mad = median - 2 * mad if mad != 0 else median  # 避免MAD为0的极端情况
    cluster_df['is_low_mad'] = duration_list < min(threshold_mad, basic_threshold)

    # 指标3：改进版Z-score（鲁棒Z-score < -2 判定为低值）
    m_z_score = modified_z_score(duration_list.values)
    cluster_df['z_score'] = m_z_score
    cluster_df['is_low_z'] = m_z_score < -2
    cluster_df['is_low_m_3sigma'] = cluster_df['z_score'] < -2  # 与is_low_z逻辑一致，保留兼容

    # 指标4：IQR下限（箱线图规则，适合非正态分布）
    q1 = duration_list.quantile(0.25)
    q3 = duration_list.quantile(0.75)
    iqr = q3 - q1
    threshold_iqr = q1 - 1.5 * iqr  # 箱线图异常值下限
    cluster_df['is_low_iqr'] = duration_list < min(threshold_iqr, basic_threshold)

    # 指标5: 滑动窗口阈值（适合非平稳序列，窗口均值-3σ）
    window_size = 10
    sliding_lower = sliding_window_threshold(duration_list, window_size=window_size, n=3)
    cluster_df['sliding_lower'] = sliding_lower
    cluster_df['is_low_sliding'] = duration_list < cluster_df['sliding_lower']

    # 最终判定：任意一个指标判定为低值 → 标记为显著低值
    cluster_df['is_significantly_low'] = (
            cluster_df['is_low_3sigma'] | cluster_df['is_low_mad'] |
            cluster_df['is_low_m_3sigma'] | cluster_df['is_low_iqr'] |
            cluster_df['is_low_sliding']
    )

    # 调用提取的函数来查找连续段
    segments_info = find_consecutive_segments(cluster_df)

    return cluster_df, segments_info


def filter_and_visualize_by_threshold(cluster_df: pd.DataFrame,
                                      threshold: float = 0.0):
    """
    基于固定阈值检测显著低值（简单直接的判定方式）
    参数:
    - cluster_df: 包含interval_total_duration列的DataFrame
    - threshold: 低值判定阈值（小于该值即为显著低值）
    - title: 可视化标题（仅用于日志，实际可视化在主函数统一调用）
    返回:
    - cluster_df: 新增below_threshold/is_significantly_low字段的DataFrame
    - consecutive_segments: 连续显著低值段信息列表，每个元素为包含开始时间、结束时间、总duration的dict
    """
    # 标记低于阈值的数据点
    duration_list = cluster_df['interval_total_duration']
    cluster_df['is_significantly_low'] = duration_list < threshold

    # 调用提取的函数来查找连续段
    segments_info = find_consecutive_segments(cluster_df)

    return cluster_df, segments_info


def few_shot_and_bias_cluster_analyze(json_path: str,
                                      visualize: bool = False,
                                      filtering_method: str = 'threshold',
                                      threshold: float = 3000,
                                      output_json_path: str = None):
    """
    多聚类异常检测主函数（支持统计方法/固定阈值，统一可视化）
    参数:
    - json_path: JSON数据文件路径（格式：{cluster_id: [{interval_total_duration: 值}, ...]}
    - visualize: 是否可视化每个聚类的检测结果
    - filtering_method: 过滤方法（'statistical'=统计多指标，'threshold'=固定阈值）
    - threshold: 阈值（filtering_method='threshold'时为固定阈值；'statistical'时为基础兜底阈值）
    - save_json: 是否将结果保存为JSON文件
    - output_json_path: 输出JSON文件路径（如果不指定则自动生成）
    - save_cluster_data: 是否保存cluster_df的完整数据
    返回:
    - all_results: 包含每个聚类的判定结果、索引、数值的字典
    """
    # 加载JSON数据
    with open(json_path, 'r', encoding='utf-8') as file:
        time_gap_dict = json.load(file)

    consecutive_segments = {}
    cluster_df_list = []  # 存储所有聚类的DataFrame
    cluster_titles = []  # 存储对应的标题
    # 遍历每个聚类进行分析
    for cluster_id, cluster_dict in time_gap_dict.items():
        # 转换为DataFrame并标记聚类ID
        cluster_df = pd.DataFrame(cluster_dict)
        cluster_df.name = cluster_id

        # 根据过滤方法执行检测
        if filtering_method == 'statistical':
            cluster_df, segments_info = few_shot_and_bias_statistics(
                cluster_df, basic_threshold=threshold
            )
        elif filtering_method == 'threshold':
            cluster_df, segments_info = filter_and_visualize_by_threshold(
                cluster_df, threshold=threshold
            )
        else:
            raise ValueError(f"不支持的过滤方法: {filtering_method}，仅支持'statistical'/'threshold'")

        # 添加到列表中，用于后续批量可视化
        cluster_df_list.append(cluster_df)
        cluster_titles.append(f"Cluster {cluster_id} - {filtering_method}方法检测结果")

        # 保存当前聚类的结果
        cluster_result = {
            'method': filtering_method,
            'segments_info': segments_info,  # 使用新的段信息结构
        }

        consecutive_segments[cluster_id] = cluster_result
    
    # 批量可视化所有聚类
    if visualize and cluster_df_list:
        visualize_significantly_low_points(
            cluster_df_list,
            titles=cluster_titles,
            save_dir=output_json_path
        )

    # 保存为JSON文件
    if output_json_path is not None:
        with open(output_json_path + f'consecutive_segments.json', 'w', encoding='utf-8') as f:
            json.dump(consecutive_segments, f, ensure_ascii=False, indent=2)
        print(f"\n【结果已保存】路径: {output_json_path + f'consecutive_segments.json'}")

    return consecutive_segments


def extract_dataset_info(json_path: str, train_threshold: float, test_threshold: float) -> dict:
    """
    从time_gap_data.json中提取训练集和测试集信息
    
    参数:
    - json_path: JSON数据文件路径
    - train_threshold: 训练集总时长阈值
    - test_threshold: 测试集总时长阈值
    
    返回:
    - dataset_info: 包含每个cluster的训练集和测试集信息的字典
    """
    # 加载JSON数据
    with open(json_path, 'r', encoding='utf-8') as file:
        time_gap_dict = json.load(file)

    dataset_info = {}

    # 遍历每个cluster
    for cluster_id, cluster_data_list in time_gap_dict.items():
        # 新的JSON格式：cluster_data_list是一个列表，每个元素包含start_timestamp, start_datetime, end_timestamp, end_datetime, interval_total_duration
        # 还包含below_threshold字段（由few_shot_and_bias_cluster_analyze函数添加）

        # 收集below_threshold为true的数据点
        train_candidates = []
        # 收集below_threshold为false的数据点
        test_candidates = []

        # 遍历所有数据点
        for idx, time_gap_info in enumerate(cluster_data_list):
            is_below = time_gap_info.get('below_threshold', False)
            duration = time_gap_info.get('interval_total_duration', 0)
            start_timestamp = time_gap_info.get('start_timestamp', None)
            start_datetime = time_gap_info.get('start_datetime', None)
            end_timestamp = time_gap_info.get('end_timestamp', None)
            end_datetime = time_gap_info.get('end_datetime', None)

            if is_below:
                # 添加到训练集候选
                train_candidates.append({
                    'index': idx,
                    'duration': duration,
                    'start_timestamp': start_timestamp,
                    'start_datetime': start_datetime,
                    'end_timestamp': end_timestamp,
                    'end_datetime': end_datetime
                })
            else:
                # 添加到测试集候选
                test_candidates.append({
                    'index': idx,
                    'duration': duration,
                    'start_timestamp': start_timestamp,
                    'start_datetime': start_datetime,
                    'end_timestamp': end_timestamp,
                    'end_datetime': end_datetime
                })

        # 训练集：按duration从低到高排序
        train_candidates.sort(key=lambda x: x['duration'])

        # 测试集：按duration从高到低排序
        test_candidates.sort(key=lambda x: x['duration'], reverse=True)

        # 构建训练集
        train_set = []
        train_total_duration = 0.0
        for candidate in train_candidates:
            if train_total_duration > train_threshold:
                break
            train_set.append(candidate)
            train_total_duration += candidate['duration']

        # 构建测试集
        test_set = []
        test_total_duration = 0.0
        for candidate in test_candidates:
            if test_total_duration > test_threshold:
                break
            test_set.append(candidate)
            test_total_duration += candidate['duration']

        # 保存当前cluster的信息
        dataset_info[cluster_id] = {
            'train_set': train_set,
            'train_total_duration': train_total_duration,
            'test_set': test_set,
            'test_total_duration': test_total_duration,
            'train_candidates_count': len(train_candidates),
            'test_candidates_count': len(test_candidates)
        }

    return dataset_info


def merge_dataframes_by_timestamp(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    合并两个DataFrame，基于timestamp列进行内连接

    参数:
    - df1: 第一个DataFrame，必须包含timestamp列
    - df2: 第二个DataFrame，必须包含timestamp列

    返回:
    - merged_df: 合并后的DataFrame，只包含两个DataFrame都存在的timestamp

    说明:
    - 使用timestamp列作为键进行内连接
    - 打印df1中未匹配的timestamp数量
    - 打印df2中未匹配的timestamp数量
    """
    import pandas as pd

    # 检查timestamp列是否存在
    if 'timestamp' not in df1.columns:
        raise ValueError("df1中不存在timestamp列")
    if 'timestamp' not in df2.columns:
        raise ValueError("df2中不存在timestamp列")

    # 获取两个DataFrame的timestamp集合
    timestamps_df1 = set(df1['timestamp'].tolist())
    timestamps_df2 = set(df2['timestamp'].tolist())

    # 找出未匹配的timestamp
    unmatched_df1 = timestamps_df1 - timestamps_df2
    unmatched_df2 = timestamps_df2 - timestamps_df1

    # 打印未匹配信息
    print(f"df1中未匹配的timestamp数量: {len(unmatched_df1)}")
    if len(unmatched_df1) > 0:
        print(f"df1中未匹配的timestamp示例: {list(unmatched_df1)[:5]}")

    print(f"df2中未匹配的timestamp数量: {len(unmatched_df2)}")
    if len(unmatched_df2) > 0:
        print(f"df2中未匹配的timestamp示例: {list(unmatched_df2)[:5]}")

    print(f"匹配的timestamp数量: {len(timestamps_df1 & timestamps_df2)}")

    # 使用merge进行内连接
    merged_df = pd.merge(df1, df2, on='timestamp', how='inner')

    return merged_df

def get_multi_cluster_datasets(data_df: pd.DataFrame, cluster_id_list: list, dataset_config: dict, save_dir: str, consecutive_segments: dict):
    """
    根据聚类ID列表生成多个聚类的数据集
    
    参数:
    - data_df: 原始数据DataFrame
    - cluster_id_list: 聚类ID列表
    - dataset_config: 数据集配置字典，包含训练集和测试集阈值
    - save_dir: 保存目录
    - consecutive_segments: 连续段信息字典
    
    返回:
    - all_results: 所有聚类的结果字典
    """
    print(f"\n【开始处理多聚类数据集】聚类数: {len(cluster_id_list)}")
    
    all_results = {}
    success_count = 0
    fail_count = 0
    
    for idx, cluster_id in enumerate(cluster_id_list, 1):
        try:
            print(f"[{idx}/{len(cluster_id_list)}] 处理聚类 {cluster_id}...", end=' ')
            
            # 检查聚类ID是否存在
            if cluster_id not in consecutive_segments:
                print(f"❌ 聚类ID {cluster_id}不存在")
                fail_count += 1
                continue
            
            # 生成数据集
            result, train_data_df, val_data_df, test_data_df = get_dataset_df(
                consecutive_segments=consecutive_segments[cluster_id]['segments_info'],
                data_df=data_df,
                low_threshold=dataset_config['train_set_threshold'],
                test_dict=dataset_config['test_set_config']
            )
            
            # 创建保存目录
            dataset_dir = os.path.join(save_dir, f'/dataset/cluster_{cluster_id}_dataset/')
            os.makedirs(dataset_dir, exist_ok=True)
            
            # 保存结果字典
            with open(os.path.join(dataset_dir, 'dataset_info.json'), 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # 保存数据集
            train_data_df.to_csv(os.path.join(dataset_dir, f'train_data_cluster_{cluster_id}.csv'), index=False)
            val_data_df.to_csv(os.path.join(dataset_dir, f'val_data_cluster_{cluster_id}.csv'), index=False)
            test_data_df.to_csv(os.path.join(dataset_dir, f'test_data_cluster_{cluster_id}.csv'), index=False)
            
            print(f"✅ 训练:{len(train_data_df)} 验证:{len(val_data_df)} 测试:{len(test_data_df)}")
            
            all_results[cluster_id] = result
            success_count += 1
            
        except Exception as e:
            print(f"❌ 错误: {str(e)}")
            fail_count += 1
            continue
    
    print(f"【完成】成功: {success_count} 失败: {fail_count}\n")
    return all_results


if __name__ == '__main__':
    # 示例：使用阈值过滤方法分析洗衣机频率聚类数据
    ANALYZE_DIR = r'./cluster_data/dbscan_result/washing_machine_freq/0.1_20_low_freq_bilistm/'
    AGGREGATE_FILE = 'E:/datasets/NILM/uk_dale/house_1/channel_1.dat'
    APPLIANCES_FILE = 'E:/datasets/NILM/uk_dale/house_1/channel_5.dat'

    print("===== 开始执行阈值过滤分析 =====")
    consecutive_segments = few_shot_and_bias_cluster_analyze(
        json_path=ANALYZE_DIR + 'time_gap_data.json',
        visualize=True,  # 开启可视化
        filtering_method='threshold',  # 可选：'statistical'（统计多指标）/'threshold'（固定阈值）
        threshold=3600,
        output_json_path=ANALYZE_DIR
    )

    print('连续段已经分析完毕')


    with open(ANALYZE_DIR + 'consecutive_segments.json', 'r', encoding='utf-8') as f:
        consecutive_segments = json.load(f)
    
    aggregate_df = pd.read_csv(
        AGGREGATE_FILE,
        sep='\s+',  # 空格分隔
        header=None,  # 无表头
        names=['timestamp', 'aggregate']  # 添加列名
    )
    
    # 读取appliances文件
    appliances_df = pd.read_csv(
        APPLIANCES_FILE,
        sep='\s+',  # 空格分隔
        header=None,  # 无表头
        names=['timestamp', 'appliance']  # 添加列名
    )
    
    merged_df = merge_dataframes_by_timestamp(aggregate_df, appliances_df)

    cluster_id_list = ['4', '5', '6', '7', '8', '9']
    dataset_config = {
        'train_set_threshold': 3600,
        'test_set_config': {
            'test_method': 'total',
            'total': {
                'time_unit': 'months',
                'time_length': 5
            },
            'duration': {
                'threshold': 60000
            }
        }
    }

    get_multi_cluster_datasets(
        data_df=merged_df,
        cluster_id_list=cluster_id_list,
        consecutive_segments=consecutive_segments,
        dataset_config=dataset_config,
        save_dir=ANALYZE_DIR
    )