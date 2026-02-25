import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

visualize_config = {
    'max_y_value': 800.0,
    'title': 'Data Visualization',
    'width': 800,
    'height': 600,
    'dpi': 100
}


def general_data_print_function(selected_data, title='Data Visualization', max_y_value=800.0):
    """
    通用数据可视化函数

    :param selected_data: 需要可视化的数据
    :param title: 图表标题
    :param max_y_value: Y轴最大值
    :return: None
    """
    # 可视化
    # 将像素转换为英寸（matplotlib使用英寸作为单位）
    fig_width = visualize_config['width'] / visualize_config['dpi']
    fig_height = visualize_config['height'] / visualize_config['dpi']

    # 可视化，设置指定尺寸
    plt.figure(figsize=(fig_width, fig_height), dpi=visualize_config['dpi'])

    # 如果数据有多列，可视化所有列
    for column in selected_data.columns:
        # 修复：将 Pandas 索引和数据转换为 numpy 数组
        # 解决 Matplotlib 与 Pandas 索引兼容性问题
        x_values = selected_data.index.to_numpy()
        y_values = selected_data[column].to_numpy()
        plt.plot(x_values, y_values, label=column)

    plt.xlabel('Timestamp')
    plt.ylabel('Active Power')
    plt.title(title)
    plt.ylim(top=max_y_value, bottom=0)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"可视化数据范围: 从索引 {selected_data.index[0]} 到 {selected_data.index[-1]}")
    print(f"实际数据点数量: {len(selected_data)}")


def visualize_first_active_data(file_path: str, threshold: float = 10.0, length: int = 1000,
                                max_y_value: float = 800, title='Data Visualization'):
    """
    设置第一列为index，并且找到第二列大于threshold的值，
    然后往后读取length个值，并且使用可视化工具对其进行可视化

    :param file_path: 数据文件路径
    :param threshold: 阈值，用于筛选第二列的值
    :param length: 要读取和可视化的数据点数量
    :param max_y_value: Y轴最大值
    :param title: 图表标题
    :return: tuple (df, start_time) - 数据框和开始时间
    """
    # 读取数据，第一列作为索引
    df = pd.read_csv(file_path, index_col=0, sep=' ', header=None)
    df.index = pd.to_datetime(df.index, unit='s')

    # 获取第二列数据
    second_column = df.iloc[:, 0]  # 第一列是索引，所以第二列是第0列

    # 找到第二列大于threshold的值的索引
    above_threshold_indices = second_column[second_column > threshold].index

    start_time = None
    if len(above_threshold_indices) > 0:
        # 获取第一个大于threshold的值的索引位置
        start_index = df.index.get_loc(above_threshold_indices[0])
        start_time = df.index[start_index]

        # 计算结束位置
        end_index = min(start_index + length, len(df))

        # 提取数据
        selected_data = df.iloc[start_index:end_index]

        # 调用通用可视化函数
        visualization_title = f'{title} (Threshold: {threshold}, Length: {len(selected_data)})'
        general_data_print_function(selected_data, visualization_title, max_y_value)

        print(f"开始时间: {start_time}")
    else:
        print(f"没有找到第二列大于阈值 {threshold} 的数据")

    return df, start_time  # 返回数据框和开始时间


def visualize_data_by_time(file_path: str, start_time: str, end_time: str, max_y_value: float = 800,
                           title='Data Visualization'):
    """
    根据指定的时间范围可视化数据

    :param file_path: 数据文件路径
    :param start_time: 开始时间
    :param end_time: 结束时间
    :param max_y_value: Y轴最大值
    :param title: 图表标题
    :return: None
    """
    df = pd.read_csv(file_path, index_col=0, sep=' ', header=None)
    df.index = pd.to_datetime(df.index, unit='s')

    # 将输入的时间字符串转换为datetime对象
    start_datetime = pd.to_datetime(start_time)
    end_datetime = pd.to_datetime(end_time)

    # 根据时间范围筛选数据
    selected_data = df[(df.index >= start_datetime) & (df.index <= end_datetime)]

    if len(selected_data) > 0:
        # 通过调用general_print_function实现可视化
        visualization_title = f'{title} (Time Range: {start_time} to {end_time})'
        general_data_print_function(selected_data, visualization_title, max_y_value)
    else:
        print(f"在指定时间范围 {start_time} 到 {end_time} 内没有找到数据")

    return selected_data


# if __name__ == '__main__':
#     # file_path = './process_dataset/experiment_dataset/BERT4NILM/aircondition/aircondition_freeze_low.dat'
#     # df, start_time = visualize_first_active_data(file_path, threshold=10.0, length=500, max_y_value=300, title='aircondition freeze low')
#     df = pd.read_csv(r'dataset_by_day/Air-condition/processed_peek_data_20250815.csv')
#     filtered_data = df['active power']
#     # 直接可视化所有数据
#     segment_df = pd.DataFrame({'active power': filtered_data[140000:145000]})
#     segment_df.index = range(len(segment_df))
#     general_data_print_function(
#         selected_data=segment_df,
#         title='Air Condition Segment',
#         max_y_value=300.0
#     )


def visualize_gap_segments(file_path: str, gap: list, max_y_value: float = 800,
                           title: str = 'Gap Segment Visualization'):
    """
    根据gap数组分割数据并分别可视化每个片段

    :param file_path: 数据文件路径
    :param gap: 分割点数组
    :param max_y_value: Y轴最大值
    :param title: 图表标题基础
    :return: None
    """
    # 读取数据
    df = pd.read_csv(file_path)
    filtered_data = df['active power']

    # 遍历gap数组，可视化每个片段
    for i in range(len(gap) - 1):
        start_idx = gap[i]
        end_idx = gap[i + 1]

        # 截取数据片段
        segment_data = filtered_data.iloc[start_idx:end_idx]

        # 创建片段DataFrame用于可视化函数
        segment_df = pd.DataFrame({'active power': segment_data})
        segment_df.index = range(len(segment_df))

        # 可视化当前片段
        segment_title = f"{title} - Segment {i + 1} ({start_idx} to {end_idx})"
        general_data_print_function(
            selected_data=segment_df,
            title=segment_title,
            max_y_value=max_y_value
        )



def visualize_all_csv_in_folder(folder_path: str, max_y_value: float = 800, title_prefix: str = 'CSV Data'):
    """
    扫描文件夹下所有CSV文件，然后逐个可视化，当用户回车的时候可视化下一个CSV文件

    :param folder_path: 文件夹路径
    :param max_y_value: Y轴最大值
    :param title_prefix: 图表标题前缀
    :return: None
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在")
        return

    # 扫描文件夹下所有CSV文件
    csv_files = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.csv'):
            csv_files.append(os.path.join(folder_path, filename))

    # 检查是否找到CSV文件
    if not csv_files:
        print(f"错误：在文件夹 '{folder_path}' 中未找到CSV文件")
        return

    # 按文件名排序
    csv_files.sort()

    print(f"找到 {len(csv_files)} 个CSV文件：")
    for i, csv_file in enumerate(csv_files, 1):
        print(f"{i}. {os.path.basename(csv_file)}")

    # 逐个可视化
    for i, csv_file in enumerate(csv_files, 1):
        filename = os.path.basename(csv_file)
        print(f"\n[{i}/{len(csv_files)}] 正在可视化：{filename}")

        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)

            # 检查文件是否有数据
            if len(df) == 0:
                print(f"警告：文件 '{filename}' 为空")
                continue

            # 检查是否有 'active power' 列
            if 'cleaned_power' in df.columns:
                data = df['cleaned_power']
            elif len(df.columns) > 0:
                # 如果没有 'active power' 列，使用第一列
                data = df.iloc[:, 0]
                print(f"警告：文件 '{filename}' 没有 'cleaned_power' 列，使用第一列 '{df.columns[0]}'")
            else:
                print(f"警告：文件 '{filename}' 没有数据列")
                continue

            # 创建用于可视化的DataFrame
            visualize_df = pd.DataFrame({'Data': data})
            visualize_df.index = range(len(visualize_df))

            # 生成图表标题
            chart_title = f"{title_prefix} - {filename}"

            # 可视化数据
            general_data_print_function(
                selected_data=visualize_df,
                title=chart_title,
                max_y_value=max_y_value
            )

            # 等待用户回车
            if i < len(csv_files):
                input("按回车键继续查看下一个文件...")
            else:
                print("所有文件已可视化完成！")

        except Exception as e:
            print(f"错误：处理文件 '{filename}' 时发生错误：{str(e)}")
            if i < len(csv_files):
                input("按回车键继续查看下一个文件...")
            continue




def visualize_npy_data(npy_file_path: str, z_index: int = 0, max_y_value: float = None, title_prefix: str = 'NPY Data'):
    """
    处理npy文件数据，可视化指定z列的所有x列数据
    - 支持3D数据：(x,y,z)形状
    - 支持2D数据：自动升维为(x,y,1)，在z=0时处理

    :param npy_file_path: npy文件路径
    :param z_index: 指定的z列索引，默认0
    :param max_y_value: Y轴最大值，None表示不限制
    :param title_prefix: 图表标题前缀
    :return: None
    """
    # 检查文件是否存在
    if not os.path.exists(npy_file_path):
        print(f"错误：文件 '{npy_file_path}' 不存在")
        return

    # 检查文件扩展名
    if not npy_file_path.lower().endswith('.npy'):
        print(f"错误：文件 '{npy_file_path}' 不是.npy文件")
        return

    try:
        # 加载npy文件
        data = np.load(npy_file_path)

        # 检查数据维度并处理
        if len(data.shape) == 2:
            print(f"检测到二维数据，正在升维为三维数据...")
            print(f"原始数据形状: {data.shape}")
            # 升维为(x,y,1)
            data = np.expand_dims(data, axis=2)
            print(f"升维后数据形状: {data.shape}")
        elif len(data.shape) != 3:
            print(f"错误：文件 '{npy_file_path}' 不是二维或三维数据")
            print(f"实际数据形状: {data.shape}")
            return

        # 获取数据形状
        x_dim, y_dim, z_dim = data.shape
        print(f"加载成功！数据形状: ({x_dim}, {y_dim}, {z_dim})")
        print(f"x维度: {x_dim}, y维度: {y_dim}, z维度: {z_dim}")

        # 检查z_index是否有效
        if z_index < 0 or z_index >= z_dim:
            print(f"错误：z_index {z_index} 超出范围 [0, {z_dim-1}]")
            return

        print(f"正在可视化 z={z_index} 列的所有 x 列数据")

        # 逐个遍历x列
        for x in range(x_dim):
            print(f"\n正在可视化 x={x} 列的数据...")

            # 提取数据：对于指定的z列，获取x列的所有y值
            y_data = data[x, :, z_index]

            # 创建用于可视化的DataFrame
            visualize_df = pd.DataFrame({'Value': y_data})
            visualize_df.index = range(len(visualize_df))

            # 生成图表标题
            chart_title = f"{title_prefix} - z={z_index}, x={x}"

            # 可视化数据
            if max_y_value is None:
                general_data_print_function(
                    selected_data=visualize_df,
                    title=chart_title
                )
            else:
                general_data_print_function(
                    selected_data=visualize_df,
                    title=chart_title,
                    max_y_value=max_y_value
                )

            # 等待用户回车
            if x < x_dim - 1:
                input("按回车键继续查看下一个x列...")
            else:
                print("所有x列数据已可视化完成！")

    except Exception as e:
        print(f"错误：处理文件 '{npy_file_path}' 时发生错误：{str(e)}")
        return


# 主程序示例
def main():
    """
    主程序示例函数，通过配置项指定调用的功能
    """
    # 配置选项
    config = {
        'function': 'npy_3d',  # 可选值: 'csv_folder', 'npy_3d'

        # CSV文件夹可视化配置
        'csv_folder': {
            'folder_path': r'ukdale_disaggregate/clasp_seg/washing_machine/data/',
            'max_y_value': 2500,
            'title_prefix': 'Air Condition Data'
        },

        # NPY三维数据可视化配置
        'npy_3d': {
            # 'npy_file_path': r'elec_feature_analyze/time_clustering/cluster_data/washing_machine/data_fusion.npy',
            'npy_file_path': 'elec_feature_analyze/time_clustering/cluster_data/washing_machine/bilstm_ae_features_cleaned_power_64_dim.npy',
            'max_y_value': 1,
            'z_index': 0,
            'title_prefix': 'NPY 3D Data'
        }
    }

    # 根据配置执行相应的功能
    if config['function'] == 'csv_folder':
        print("执行功能：可视化文件夹下所有CSV文件")
        csv_config = config['csv_folder']
        visualize_all_csv_in_folder(
            folder_path=csv_config['folder_path'],
            max_y_value=csv_config['max_y_value'],
            title_prefix=csv_config['title_prefix']
        )

    elif config['function'] == 'npy_3d':
        print("执行功能：可视化NPY三维数据")
        npy_config = config['npy_3d']
        visualize_npy_data(
            npy_file_path=npy_config['npy_file_path'],
            z_index=npy_config['z_index'],
            max_y_value=npy_config['max_y_value'],
            title_prefix=npy_config['title_prefix']
        )

    else:
        print(f"错误：未知的功能类型 '{config['function']}'")
        print("可选功能类型：'csv_folder', 'npy_3d'")


if __name__ == '__main__':
    main()

