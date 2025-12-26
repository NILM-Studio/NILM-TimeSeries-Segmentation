import os,re
import traceback

import matplotlib.pyplot as plt
import pandas as pd
from sktime.detection.clasp import find_dominant_window_sizes
import clasp
import fluss

WINDOW_SIZE = 20
SHOW_SEG_RES = False


def save_changepoint_info(df, cps, output_file):
    """
    Save information about changepoints to a CSV file

    Args:
        df: DataFrame containing the time series data
        cps: List of changepoint indices from FLUSS algorithm
        output_file: Path to output CSV file
    """
    # Create a DataFrame to store changepoint information
    changepoint_data = []
    colum = cps.iloc[:, 0]

    # Add information for each changepoint
    for cp_index in cps.iloc[:, 0]:
        if cp_index < len(df):
            # Get the row corresponding to the changepoint
            row_data = df.iloc[cp_index].copy()
            # Add the index/position information
            row_data['changepoint_index'] = cp_index
            changepoint_data.append(row_data)

    # Convert to DataFrame
    cp_df = pd.DataFrame(changepoint_data)

    # Save to CSV
    cp_df.to_csv(output_file, index=False)
    print(f"Saved {len(cps)} change-points to {output_file}")


def visualize_and_seg_csv_files(folder_path, output_folder):
    """
    读取指定文件夹下的所有.csv文件为DataFrame并且逐个可视化

    Args:
        :param folder_path:
        :param output_folder:
    """
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    if not csv_files:
        print(f"在 {folder_path} 中未找到CSV文件")
        return

    print(f"找到 {len(csv_files)} 个CSV文件:")
    for i, file in enumerate(csv_files):
        print(f"[{i}] {file}")

    alg = input("输入要使用的语义分割算法: 输入f使用FLUSS, 输入c使用CLASP ")
    # 逐个处理每个CSV文件
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        print(f"\n正在处理文件: {file}")

        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 检查是否有足够的列
            if len(df.columns) < 2:
                print(f"文件 {file} 列数不足，跳过")
                continue

            # 设置图表
            plt.figure(figsize=(12, 6))
            tsp = df.iloc[:, 0].values
            pw = df.iloc[:, 1].values
            plt.plot(tsp, pw, linewidth=1)
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title(f'Visualization of {file}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            # 等待用户输入
            while True:
                output_file = os.path.join(output_folder, f"Changepoints_{file}")
                if os.path.exists(output_file):
                    print(f"File {file} has been processed and saved to {output_file}, SKIP")
                    continue
                user_input = input("请输入命令 (数字n调用分割算法，'e'退出程序，'s'跳过到下一个文件): ")

                if user_input.lower() == 'e':
                    print("退出程序")
                    return
                elif user_input.lower() == 's':
                    print("跳过到下一个文件")
                    continue
                else:
                    try:
                        n_regimes = int(user_input)
                        # 调用fluss函数
                        ts = df.iloc[:, 1]  # 提取第2列数据作为时间序列
                        if alg == 'c':
                            print(f"调用clasp函数，n_regimes={n_regimes}")
                            window_size = find_dominant_window_sizes(ts)
                            if window_size is None:
                                window_size = WINDOW_SIZE
                                print(f"Using default window size: {window_size}")
                            else:
                                print("Dominant Period", window_size)
                            ts, cps = clasp.clasp_nilm(ts, 20, n_regimes)
                            pass
                            # ts, cps = fluss.clasp(ts, window_size=50, n_regimes=n_regimes, excl_factor=1)
                        elif alg == 'f':
                            print(f"调用fluss函数，n_regimes={n_regimes}")
                            ts, cps = fluss.fluss(ts, window_size=WINDOW_SIZE, n_regimes=n_regimes,
                                                  excl_factor=1)
                        else:
                            raise ValueError("Invalid algorithm specified")
                        if len(cps) == 0:
                            print("No changepoints found")
                        else:
                            save_changepoint_info(df, cps, output_file)
                        break
                    except ValueError:
                        print("非法输入，请输入一个数字、'exit'或'skip'")

        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")
            print(f"错误类型: {type(e).__name__}")
            print(f"详细错误信息:")
            traceback.print_exc()
            continue

    print("所有文件处理完成")



if __name__ == '__main__':
    # time_series = pd.read_csv('../process_dataset/Air-condition/Air_condition.csv')
    # # 确保 'active power' 列是数值类型，非数值数据会被转换为 NaN
    # time_series['active power'] = pd.to_numeric(time_series['active power'], errors='coerce')
    #
    # # 删除或填充 NaN 值（这里选择删除）
    # ts = time_series.dropna(subset=['active power'])
    # ts = time_series.head(6000)
    # ts = ts['active power'].values  # 提取数值数组
    #
    # fluss.fluss(ts, window_size=300, n_regimes=10, excl_factor=1)
    visualize_and_seg_csv_files('../../ukdale_disaggregate/process_data/washing_machine_channel_5',
                                '../../ukdale_disaggregate/cps/washing_machine')
    # folder_path = "../../ukdale_disaggregate/process_data/washing_machine_channel_5/"
    # rename_channel_files(folder_path, "channel_5", "Washing_Machine")
