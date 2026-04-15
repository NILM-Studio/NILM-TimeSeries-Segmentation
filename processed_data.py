import os
import shutil
import pandas as pd
import visualize as vis
import numpy as np


def move_datafile_to_dataset(main_file: str, app_file: str, dataset_path, channel: int):
    """
    检查dataset_path下是否存在channel1_org.dat，如果不存在则将channel1.dat复制并重命名为channel1_org.dat
    将main_file复制到dataset_path下并重命名为channel1.dat
    将app_file复制到dataset_path下并根据参数channel重命名

    :param main_file: 主文件路径
    :param app_file: 应用文件路径
    :param dataset_path: 数据集路径
    :param channel: 通道号
    :return: None
    """
    # 构建目标文件路径
    channel1_org_path = os.path.join(dataset_path, 'channel_1_org.dat')
    channel1_path = os.path.join(dataset_path, 'channel_1.dat')
    channel_app_path = os.path.join(dataset_path, f'channel_{channel}.dat')

    print(f"开始处理文件移动到数据集: {dataset_path}")
    print(f"主文件: {main_file}")
    print(f"应用文件: {app_file}")
    print(f"目标通道: {channel}")

    # 检查channel1_org.dat是否存在
    if not os.path.exists(channel1_org_path):
        print(f"文件 {channel1_org_path} 不存在，检查源文件...")
        # 检查源文件channel1.dat是否存在
        if os.path.exists(channel1_path):
            # 复制并重命名文件
            shutil.copy2(channel1_path, channel1_org_path)
            print(f"已复制 {channel1_path} 到 {channel1_org_path}")
        else:
            print(f"警告: 源文件 {channel1_path} 不存在")
    else:
        print(f"文件 {channel1_org_path} 已存在，跳过复制")

    # 复制main_file到dataset_path并重命名为channel1.dat
    if os.path.exists(main_file):
        shutil.copy2(main_file, channel1_path)
        print(f"已复制主文件 {main_file} 到 {channel1_path}")
    else:
        raise FileNotFoundError(f"主文件 {main_file} 不存在")

    # 复制app_file到dataset_path并重命名为channel_{channel}.dat
    if os.path.exists(app_file):
        shutil.copy2(app_file, channel_app_path)
        print(f"已复制应用文件 {app_file} 到 {channel_app_path}")
    else:
        raise FileNotFoundError(f"应用文件 {app_file} 不存在")

    print("文件处理完成！")


def calculate_status_active_times(app_df: pd.DataFrame, target_status: int):
    """
    计算某个状态工作了几次，app_df为一个index为datetime，包含power  active power  on/off status 和status列的DataFrame,
    某一段连续的status==target_status视作一次active，统计target_status了多少次，共计时常是多少，并且将每次active的开始、结束
    时间和持续时间长度（以s为单位）存入一个数组中返回
    :return: (次数, 总时长, 每次活动的详细信息列表)
    """
    # 检查必要的列是否存在
    if 'status' not in app_df.columns:
        raise ValueError("DataFrame中必须包含'status'列")

    # 特殊情况：target_status为-1时，统计所有非0的数据段
    if target_status == -1:
        print("计算所有非0状态的活动时间段")
        # 初始化变量
        active_count = 0  # 活动次数
        total_duration = 0  # 总时长
        active_periods = []  # 每次活动的详细信息 [(开始时间, 结束时间, 持续时间), ...]

        # 初始化状态跟踪变量
        in_nonzero_status = False
        start_time = None

        # 遍历DataFrame的每一行
        for idx, row in app_df.iterrows():
            current_status = row['status']

            # 如果当前状态是非0状态且之前是0状态，则开始一个新的活动期
            if current_status != 0 and not in_nonzero_status:
                in_nonzero_status = True
                start_time = idx

            # 如果当前状态是0状态且之前是非0状态，则结束当前活动期
            elif current_status == 0 and in_nonzero_status:
                in_nonzero_status = False
                end_time = idx

                duration = (end_time - start_time).total_seconds()

                # 更新统计信息
                active_count += 1
                total_duration += duration
                active_periods.append((start_time, end_time, duration))

        # 处理最后一条记录仍处于非0状态的情况
        if in_nonzero_status:
            end_time = app_df.index[-1]
            duration = (end_time - start_time).total_seconds()
            active_count += 1
            total_duration += duration
            active_periods.append((start_time, end_time, duration))

        print(f"所有非0状态的活动统计:")
        print(f"  活动次数: {active_count}")
        print(f"  总时长: {total_duration} 秒 ({total_duration / 3600:.2f} 小时)")
        print(f"  平均每次持续时间: {total_duration / active_count if active_count > 0 else 0:.2f} 秒")

        return active_count, total_duration, active_periods

    print(f"计算状态{target_status} active的时间段")
    # 初始化变量
    active_count = 0  # 活动次数
    total_duration = 0  # 总时长
    active_periods = []  # 每次活动的详细信息 [(开始时间, 结束时间, 持续时间), ...]

    # 初始化状态跟踪变量
    in_target_status = False
    start_time = None

    # 遍历DataFrame的每一行
    for idx, row in app_df.iterrows():
        current_status = row['status']

        # 如果当前状态是目标状态且之前不是目标状态，则开始一个新的活动期
        if current_status == target_status and not in_target_status:
            in_target_status = True
            start_time = idx

        # 如果当前状态不是目标状态且之前是目标状态，则结束当前活动期
        elif current_status != target_status and in_target_status:
            in_target_status = False
            end_time = idx

            duration = (end_time - start_time).total_seconds()

            # 更新统计信息
            active_count += 1
            total_duration += duration
            active_periods.append((start_time, end_time, duration))

    # 处理最后一条记录仍在目标状态的情况
    if in_target_status:
        end_time = app_df.index[-1]
        duration = (end_time - start_time).total_seconds()
        active_count += 1
        total_duration += duration
        active_periods.append((start_time, end_time, duration))

    print(f"状态 '{target_status}' 的活动统计:")
    print(f"  活动次数: {active_count}")
    print(f"  总时长: {total_duration} 秒 ({total_duration / 3600:.2f} 小时)")
    print(f"  平均每次持续时间: {total_duration / active_count if active_count > 0 else 0:.2f} 秒")

    return active_count, total_duration, active_periods


def clean_data_by_list(df: pd.DataFrame, periods_list):
    """
    对于某一个状态的用电器数据，将periods_list中的每个时间段的status和on/off status置为0，将apparent power和active power置为1
    :param df:
    :param periods_list: 时间段列表，每个元素为(start_time, end_time, duration)的三元组
    :return: None
    """
    print(f"需要处理的时间段数量: {len(periods_list)}")

    # 对每个时间段进行处理
    for i, (start_time, end_time, duration) in enumerate(periods_list):
        print(f"处理时间段 {i + 1}/{len(periods_list)}: {start_time} 到 {end_time} (持续时间: {duration}秒)")

        # 创建时间范围掩码
        mask = (df.index >= start_time) & (df.index <= end_time)

        # 修改对应行的数据
        modified_rows = mask.sum()
        print(f"  匹配到 {modified_rows} 行数据")

        if 'status' in df.columns:
            df.loc[mask, 'status'] = 0
        if 'on/off status' in df.columns:
            df.loc[mask, 'on/off status'] = 0
        if 'active power' in df.columns:
            df.loc[mask, 'active power'] = 1
        if 'apparent power' in df.columns:
            df.loc[mask, 'apparent power'] = 1

    # 保存处理后的数据
    print("数据清理完成！")
    return df


def clean_data(df: pd.DataFrame, status: int):
    """

    :param status:
    :param df:
    :return:
    """
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
    if 'status' in df.columns:
        df['status'] = df['status'].round()

    if 'on/off status' in df.columns:
        df['on/off status'] = df['on/off status'].round()
    active_count, total_duration, removed_periods = calculate_status_active_times(df, status)

    # removed_periods是需要删除的时间段，如果是需要留下来的数据，在接下来的步骤中，将这些时间段从removed_periods剔除
    while True:
        # 打印带下标的active_periods
        print("\n当前活动时间段列表:")
        for i, period in enumerate(removed_periods):
            print(f"[{i}] {period}")

        # 如果列表为空，退出循环
        if not removed_periods:
            print("活动时间段列表为空，退出循环。")
            break

        # 获取用户输入
        user_input = input("\n需要保留哪个下标的数据？(输入要移除的下标，输入'quit'退出): ")

        # 检查是否退出
        if user_input.lower() == 'quit':
            print("退出交互循环。")
            break

        # 尝试解析用户输入为整数
        try:
            index_to_remove = int(user_input)
            # 检查下标是否有效
            if 0 <= index_to_remove < len(removed_periods):
                removed_period = removed_periods.pop(index_to_remove)
                print(f"已移除下标为 {index_to_remove} 的元素: {removed_period}")
                # 计算剩余活动时间段的总时长
                remaining_total_duration = sum(period[2] for period in removed_periods)

                # 计算删除的百分比和剩余的百分比
                if total_duration > 0:
                    deleted_percentage = (remaining_total_duration / total_duration) * 100
                    remaining_percentage = ((total_duration - remaining_total_duration) / total_duration) * 100
                else:
                    deleted_percentage = 0
                    remaining_percentage = 0

                # 打印统计信息
                print(f"\n时长统计:")
                print(f"  原始总时长: {total_duration:.2f} 秒")
                print(f"  删除列表总时长: {remaining_total_duration:.2f} 秒")
                print(f"  剩余列表总时长: {total_duration - remaining_total_duration:.2f} 秒")
                print(f"  将会删除 {deleted_percentage:.2f}% 的数据")
                print(f"  剩余 {remaining_percentage:.2f}% 的数据")
            else:
                print(f"错误: 下标 {index_to_remove} 超出范围 [0, {len(removed_periods) - 1}]")
        except ValueError:
            print("错误: 请输入有效的整数或 'quit'")

    print(f"\n最终保留的活动时间段数量: {len(removed_periods)}")
    for i, period in enumerate(removed_periods):
        print(f"[{i}] {period}")

    clean_data_by_list(df, removed_periods)
    active_count, total_duration, periods = calculate_status_active_times(df, status)
    for i, period in enumerate(periods):
        print(f"[{i}] {period}")
    print(f'剩余的列表总时长: {total_duration:.2f} 秒')
    return df


def print_all_periods(df: pd.DataFrame, status_num: int):
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
    if 'status' in df.columns:
        df['status'] = df['status'].round()

    if 'on/off status' in df.columns:
        df['on/off status'] = df['on/off status'].round()
    for i in range(0, status_num + 1):
        active_count, total_duration, periods = calculate_status_active_times(df, i)
        for j, period in enumerate(periods):
            print(f"[{j}] {period}")
        print(f'剩余的列表总时长: {total_duration:.2f} 秒')


def downsample_data(df: pd.DataFrame, downsample_rate: int, method: str):
    """
    根据指定方法对DataFrame进行下采样

    Args:
        df: 输入的DataFrame
        downsample_rate: 下采样率，表示每多少行合并为一行
        method: 下采样方法 ('avg', 'min', 'max')

    Returns:
        下采样后的DataFrame
    """
    if method == 'avg':
        # 均值下采样
        return df.groupby(df.index // downsample_rate).mean()
    elif method == 'min':
        # 最小值下采样
        return df.groupby(df.index // downsample_rate).min()
    elif method == 'max':
        # 最大值下采样
        return df.groupby(df.index // downsample_rate).max()
    else:
        raise ValueError("method must be 'avg', 'min', or 'max'")


def filter_data(df: pd.DataFrame, window_size: int, method: str):
    """
    根据指定方法对DataFrame进行滤波处理

    Args:
        df: 输入的DataFrame
        window_size: 滤波窗口大小
        method: 滤波方法 ('moving_avg', 'median', 'lowpass')

    Returns:
        滤波后的DataFrame
    """
    # 确保索引是连续的整数索引，以便滤波操作
    df_reset = df.reset_index(drop=True)

    if method == 'moving_avg':
        # 移动平均滤波
        filtered_df = df_reset.rolling(window=window_size, min_periods=1).mean()
    elif method == 'median':
        # 中值滤波
        filtered_df = df_reset.rolling(window=window_size, min_periods=1).median()
    elif method == 'lowpass':
        # 低通滤波（使用移动平均实现简单的低通滤波效果）
        filtered_df = df_reset.rolling(window=window_size, min_periods=1).mean()
    else:
        raise ValueError("method must be 'moving_avg', 'median', or 'lowpass'")

    # 恢复原始索引
    filtered_df.index = df.index
    vis.general_data_print_function(filtered_df[:3000], title=f'Data Visualization ({method} Filtered)',
                                    max_y_value=800)
    return filtered_df


def get_active_segments(df: pd.DataFrame, threshold: float, save_file: bool = False, file_name: str = 'segment',
                        save_file_path: str = r'./ukdale_disaggregate/active', context_size: int = 300):
    # 使用第1列作为数据列
    data_column = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    # 找到大于阈值的数据点
    above_threshold = df[data_column] > threshold

    # 标记连续段
    segments = []
    in_segment = False
    start_idx = None

    # 获取索引列表以便进行位置计算
    index_list = df.index.tolist()
    index_positions = {idx: pos for pos, idx in enumerate(index_list)}

    # 遍历数据识别连续段
    for idx, is_above in above_threshold.items():
        if is_above and not in_segment:
            # 开始一个新的段
            start_idx = idx
            in_segment = True
        elif not is_above and in_segment:
            # 结束当前段
            end_idx = idx

            # 计算包含上下文的起始和结束位置
            start_pos = max(0, index_positions[start_idx] - context_size)
            end_pos = min(len(index_list) - 1, index_positions[end_idx] + context_size)

            # 获取包含上下文的段
            context_start_idx = index_list[start_pos]
            context_end_idx = index_list[end_pos]
            segment_df = df.loc[context_start_idx:context_end_idx].copy()
            # 剔除工作时间小于100的段
            if end_idx - start_idx < 100:
                continue
            segments.append((segment_df, start_idx, end_idx))
            in_segment = False
            print(f'Get Segment that start at {start_idx} and end at {end_idx}')

    # 处理最后一个段延续到数据末尾的情况
    if in_segment:
        end_idx = df.index[-1]

        # 计算包含上下文的起始和结束位置
        start_pos = max(0, index_positions[start_idx] - context_size)
        end_pos = min(len(index_list) - 1, index_positions[end_idx] + context_size)

        # 获取包含上下文的段
        context_start_idx = index_list[start_pos]
        context_end_idx = index_list[end_pos]
        segment_df = df.loc[context_start_idx:context_end_idx].copy()
        segments.append((segment_df, start_idx, end_idx))

    # 如果需要保存文件
    if save_file:
        os.makedirs(save_file_path, exist_ok=True)
        for i, (segment, start_index, end_index) in enumerate(segments):
            # 计算持续时间（假设索引是时间戳）
            try:
                # 使用实际的时间戳而不是索引值
                start_timestamp = df.loc[start_index, df.columns[0]]  # 获取第0列的时间戳
                end_timestamp = df.loc[end_index, df.columns[0]]  # 获取第0列的时间戳

                # 计算持续时间（假设时间戳是秒单位）
                duration = (end_timestamp - start_timestamp)

                # 将时间戳转换为 YYYY-MM-DD HH:MM:SS 格式
                start_time_str = pd.to_datetime(start_timestamp, unit='s').strftime('%Y-%m-%d_%H-%M-%S')
                end_time_str = pd.to_datetime(end_timestamp, unit='s').strftime('%Y-%m-%d_%H-%M-%S')
                filename = os.path.join(save_file_path, f"{file_name}_{start_time_str}_{end_time_str}_{duration:.0f}.csv")
            except:
                # 如果无法计算时间差，使用索引值
                duration = end_index - start_index
                filename = os.path.join(save_file_path, f"{file_name}_{start_index}_{end_index}_{duration}.csv")
            segment.to_csv(filename)

    # 只返回DataFrame列表，不包含时间信息
    return segments


if __name__ == '__main__':
    # main_file = r'.\experiment_dataset\BERT4NILM\microwave\test\mains_sterilize.dat'
    # app_file = r'.\experiment_dataset\BERT4NILM\microwave\test\microwave_sterilize.dat'
    # dataset_path = r'E:\datasets\NILM\uk_dale\house_2'
    # channel = 15
    # move_datafile_to_dataset(main_file, app_file, dataset_path, channel)

    file_path = r'./process_dataset/WashMachine/WashMachine.csv'
    df = pd.read_csv(file_path, index_col=0)
    # df_after_process = clean_data(df, 2)
    # df_after_process.to_csv(file_path, index=True)
    # print(f"已保存处理后的数据到: {file_path}")
    print_all_periods(df, 4)

    # df = pd.read_csv(r'.\Air-condition\Air_condition.csv')
    # df = filter_data(df, 10, 'moving_avg')
    # df.to_csv(r'.\Air-condition\Air_condition_10_avg_filter.csv')

    # 获取原始.dat序列并且找到工作的
    # df = pd.read_csv(r'E:\datasets\NILM\uk_dale\house_1\channel_5.dat', delimiter=' ')
    # print(df)
    # get_active_segments(df, 10, save_file=True, save_file_path=r'.\ukdale_disaggregate\active\washing_machine',
    #                     context_size=100)
    # exit()
