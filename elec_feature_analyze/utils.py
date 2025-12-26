import os
import re
import shutil
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import glob


def get_file_by_time(file_dir, start_time: str = None, end_time: str = None):
    """
    根据时间范围筛选文件

    :param file_dir: 文件目录路径
    :param start_time: 开始时间范围
    :param end_time: 结束时间范围
    :return: 符合时间条件的文件信息列表
    """
    # 处理字符串时间参数
    if isinstance(start_time, str):
        start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    if isinstance(end_time, str):
        end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    # 获取目录中的所有文件
    files = os.listdir(file_dir)

    time_info_list = []

    for file in files:
        if file.endswith('.csv'):
            # 提取时间戳
            print(f'正在处理文件: {file}')
            pattern = r'(\d{8}_\d{6})_(\d{8}_\d{6})'
            match = re.search(pattern, file)

            if match:
                start_str = match.group(1)
                end_str = match.group(2)

                # 转换为datetime对象
                file_start_time = datetime.strptime(start_str, '%Y%m%d_%H%M%S')
                file_end_time = datetime.strptime(end_str, '%Y%m%d_%H%M%S')

                # 判断条件：当start_time和end_time均为None时，或者满足时间范围条件时才加入列表
                if (start_time is None and end_time is None) or \
                        (start_time is not None and end_time is not None and
                         file_start_time >= start_time and file_end_time <= end_time):
                    print(f"文件: {file} 符合时间条件")
                    time_info_list.append({
                        'filename': file,
                        'start_time': file_start_time,
                        'end_time': file_end_time
                    })

    # 打印文件总数
    print(f"找到符合条件的文件总数: {len(time_info_list)}")

    # 如果有符合条件的文件，询问用户是否复制
    if len(time_info_list) > 0:
        user_input = input("是否将这些文件复制到 flite_files 文件夹？(y/n): ")

        if user_input.lower() == 'y':
            # 创建 flite_files 文件夹
            flite_dir = os.path.join(file_dir, "flite_files")
            os.makedirs(flite_dir, exist_ok=True)

            # 复制文件
            for file_info in time_info_list:
                src_path = os.path.join(file_dir, file_info['filename'])
                dst_path = os.path.join(flite_dir, file_info['filename'])
                shutil.copy2(src_path, dst_path)
                print(f"已复制: {file_info['filename']}")

            print(f"所有文件已复制到: {flite_dir}")

    return time_info_list


def visualize_csv_files(folder_path):
    """
    遍历指定文件夹下的所有CSV文件并可视化
    用户按回车键继续查看下一个文件
    """
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not csv_files:
        print(f"在 {folder_path} 中未找到CSV文件")
        return

    print(f"找到 {len(csv_files)} 个CSV文件:")
    for i, file in enumerate(csv_files, 1):
        print(f"{i}. {os.path.basename(file)}")

    # 遍历并可视化每个CSV文件
    for file_path in csv_files:
        # try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        filename = os.path.basename(file_path)
        print(f"\n正在处理: {filename}")
        print("数据预览:")
        print(df.head())
        # 创建可视化
        plt.figure(figsize=(10, 6))
        # 如果数据列数适合绘图，则创建图表
        if not df.empty and len(df.columns) >= 2:
            x_column = df.columns[0]  # 第1列
            y_column = df.columns[1]  # 第0列
            # 绘制散点图
            plt.plot(df[x_column].values, df[y_column].values)
            plt.title(f'数据可视化 - {filename}')
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.tight_layout()
            plt.show()
        # 等待用户输入回车键
        input("按回车键继续查看下一个文件...")

        # except Exception as e:
        #     print(f"处理文件 {file_path} 时出错: {e}")
        #     input("按回车键继续查看下一个文件...")


def rename_channel_files(folder_path, prefix, new_prefix):
    """
    遍历文件夹中所有文件，将包含 prefix 的文件名替换为指定前缀

    Args:
        :param new_prefix:
        :param folder_path:
        :param prefix:
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"文件夹 {folder_path} 不存在")
        return

    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)

    # 统计重命名的文件数量
    renamed_count = 0

    for filename in files:
        # 检查文件名是否包含 channel_36
        if prefix in filename:
            # 构造新的文件名
            new_filename = filename.replace(prefix, new_prefix)

            # 构造完整的文件路径
            old_filepath = os.path.join(folder_path, filename)
            new_filepath = os.path.join(folder_path, new_filename)

            try:
                # 重命名文件
                os.rename(old_filepath, new_filepath)
                print(f"重命名: {filename} -> {new_filename}")
                renamed_count += 1
            except Exception as e:
                print(f"重命名文件 {filename} 时出错: {e}")

    print(f"共重命名了 {renamed_count} 个文件")


if __name__ == "__main__":
    file_dir = r'../ukdale_disaggregate/process_data/washing_machine_channel_5'
    # start_time = "2012-11-11 00:00:00"
    # end_time = "2013-11-11 00:00:00"
    get_file_by_time(file_dir)
    # visualize_csv_files(file_dir)
    # folder_path = "../ukdale_disaggregate/process_data/solar_thermal_pump_channel_3/"
    # rename_channel_files(folder_path, "channel_3", "Solar_Thermal_Pump")
