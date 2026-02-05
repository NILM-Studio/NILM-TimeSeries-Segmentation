import json
import os
import pandas as pd
import numpy as np

# ===================== 1. 配置项（大写常量，统一管理） =====================
ACTIVE_DIR = r"../../ukdale_disaggregate/clasp_seg/washing_machine_freq/data"
CPS_DIR = r"../../ukdale_disaggregate/clasp_seg/washing_machine_freq/label"
OUTPUT_DIR = r"./cluster_data/washing_machine_freq/"
APPLIANCE_NAME = "washing_machine_freq"
'''
可选DATA_COLUMN:
- `power`: The original power value at the timestamp.
- `cleaned_power`: The power processed by median filter at the timestamp.
- `high_freq`: The high-frequency signal component by db4 wavelet decomposition.
- `low_freq`: The low-frequency signal component by db4 wavelet decomposition.
'''
DATA_COLUMN = ['power', 'cleaned_power', 'high_freq', 'low_freq']
CSV_ENCODING = "utf-8"  # 若报编码错，可改为"gbk"或"utf-8-sig"
SAVE_NON_MATCH_FILE = True


# ===================== 2. 核心函数：以active文件为核心匹配+读取 =====================
def match_active_with_cps(label_type_filter=0):
    """
    核心逻辑：遍历active文件夹，匹配对应的CPS文件，读取active文件为DataFrame，构建匹配结果
    参数：
        label_type_filter: int, 指定要获取的label_type值，-1表示获取所有cps
    返回：
        match_results: 列表[字典]，每个字典对应一个active文件的匹配+数据信息
    """
    # 初始化匹配结果列表（以active文件为核心）
    match_results = []

    # 步骤1：遍历active文件夹下所有CSV文件
    for active_filename in os.listdir(ACTIVE_DIR):
        # 过滤：仅处理.csv文件，排除Excel临时文件（~$开头）
        if not active_filename.endswith(".csv") or active_filename.startswith("~$"):
            continue

        # 提取active文件核心信息
        active_prefix = active_filename[:-4]  # 去掉.csv后缀，如"xxx.csv"→"xxx"
        active_path = os.path.join(ACTIVE_DIR, active_filename)

        # 步骤2：匹配对应的CPS文件
        cps_target_filename = f"Changepoints_{active_prefix}.csv"
        cps_target_path = os.path.join(CPS_DIR, cps_target_filename)

        # 初始化当前active文件的匹配信息
        current_match = {
            "appliance": APPLIANCE_NAME,
            "data_file": active_filename,  # active文件名（如xxx.csv）
            "data_path": active_path,  # active文件完整路径
            "data_column": DATA_COLUMN,
            "data": None,  # 存储读取后的DataFrame
            "cps_file": None,  # 对应的CPS文件名（如Changepoint_xxx.csv）
            "cps": None,
            "cut_data": [],
            "match_status": "No_CPS",  # 匹配状态：Success/None_CPS
        }

        # 检查CPS文件是否存在
        if os.path.exists(cps_target_path):
            current_match["cps_file"] = cps_target_filename
            current_match["match_status"] = "Success"
            try:
                cps_df = pd.read_csv(cps_target_path, encoding=CSV_ENCODING)

                # 根据label_type过滤CPS数据
                if label_type_filter != -1:
                    cps_df = cps_df[cps_df['label_type'] == label_type_filter]

                current_match["cps"] = cps_df
            except Exception as e:
                print(f"Error reading CPS file: {cps_target_path}")
                print(f"Error details: {str(e)}")
                current_match["cps"] = None
        else:
            print(f"× No CPS file found: {cps_target_path},SKIP")

        # 步骤3：读取active文件为DataFrame（无论是否匹配到CPS文件都尝试读取）
        try:
            df = pd.read_csv(active_path, encoding=CSV_ENCODING)
            current_match["data"] = df
            print(f"√ Read success: {active_filename} | CPS match: {current_match['match_status']}")
        except Exception as e:
            current_match["error_msg"] = str(e)
            print(f"----///----× Read failed: {active_filename} | Error: {str(e)} ×----///----")

        # 将当前active文件的信息加入结果列表
        match_results.append(current_match)

    # 步骤4：打印匹配/读取汇总
    print("\n" + "=" * 60 + " Summary " + "=" * 60)
    total_active = len(match_results)
    match_success = sum(1 for res in match_results if res["match_status"] == "Success")
    print(f"Total active files: {total_active}")
    print(f"CPS match success: {match_success} | None match: {total_active - match_success}")

    if label_type_filter != -1:
        cps_count = sum(len(res["cps"]) if res["cps"] is not None else 0 for res in match_results)
        print(f"Filtered CPS records with label_type {label_type_filter}: {cps_count}")

    return match_results


def cutting_data_by_cps():
    """
    根据变点将数据切分为n+1段
    :return:
    """
    cut_data_list = []
    match_results = match_active_with_cps()
    print("\n\n---------------MATCH FINISHED!-------------\n\n")
    for res in match_results:
        print(f"\n\nSegmenting Data:{res['data_file']}")
        if res["match_status"] == "Success":
            cps = res["cps"]  # 变点DataFrame
            data = res["data"]

            # 清空之前的切割数据
            res["cut_data"] = []

            # 获取所有变点的时间戳
            timestamps = cps['timestamp'].tolist()  # 假设变点列名为'timestamp'
            timestamps.sort()

            # 添加起始时间戳和结束时间戳
            timestamps = [data['timestamp'].iloc[0].item()] + timestamps + [data['timestamp'].iloc[-1].item()]

            # 根据时间戳切分数据为n+1段
            for i in range(len(timestamps) - 1):
                start_time = timestamps[i]
                end_time = timestamps[i + 1]

                # 根据timestamp列筛选数据
                mask = (data['timestamp'] >= start_time) & (data['timestamp'] < end_time)
                cut_data = data[mask]
                if len(cut_data) < 10:
                    print(f"Warning: Cutting data segment {i + 1} is too short, less than 10 records")
                    continue
                print(f"Cutting data segment {i + 1}: from {start_time} to {end_time}, got {len(cut_data)} records")

                res["cut_data"].append(cut_data)
                cut_res = {
                    "data_file": res["data_file"],
                    "appliance": APPLIANCE_NAME,
                    "start_timestamp": start_time,
                    "end_timestamp": end_time,
                    "data": cut_data
                }
                cut_data_list.append(cut_res)
        elif res["match_status"] == "No_CPS":
            if SAVE_NON_MATCH_FILE and res["data"] is not None:
                print(f"No CPS file found for {res['data_file']}, output origin data directly")
                data = res["data"]
                cut_res = {
                    "data_file": res["data_file"],
                    "appliance": APPLIANCE_NAME,
                    "start_timestamp": data['timestamp'].iloc[0].item(),
                    "end_timestamp": data['timestamp'].iloc[-1].item(),
                    "data": data
                }
                res["cut_data"].append(data)
                cut_data_list.append(cut_res)
            else:
                print(f"No CPS file found for {res['data_file']}, SKIP")
                continue
        else:
            continue

    return cut_data_list, match_results


def save_file_for_cluster(extract_list=None):
    """
    将切割后的数据展平并填充为相同长度，存储为numpy数组格式以便后续聚类处理
    :param extract_list: 要提取的列名列表，默认为['power']
    :returns
    padded_array: 完成展平后的数据，维度为(n, max_len, num_features)
                 其中num_features为extract_list的长度
    lengths_array: 每个samples的长度，(n, 1)
    """
    import matplotlib.pyplot as plt

    if extract_list is None:
        extract_list = ['power']
    cut_data_list, match_results = cutting_data_by_cps()
    ts_list = []
    max_len = 0

    # 第一步：找出所有DataFrame中最长的长度
    for cut_res in cut_data_list:
        df = cut_res["data"]
        del cut_res["data"]
        ts_list.append(df)
        if len(df) > max_len:
            max_len = len(df)

    # 第二步：根据指定列的数量创建numpy数组
    n = len(ts_list)
    num_features = len(extract_list)
    padded_array = np.zeros((n, max_len, num_features))  # 特征数量根据指定列数确定

    # 创建用于记录每个df实际数据长度的数组
    lengths_array = np.zeros(n, dtype=int)

    # 第三步：对每个DataFrame进行展平和填充操作
    for i, df in enumerate(ts_list):
        # 检查DataFrame是否包含所有需要的列
        missing_cols = [col for col in extract_list if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame {i} is missing required columns: {missing_cols}")

        # 提取指定的列数据
        extracted_data = []
        for col in extract_list:
            col_data = df[col].values
            extracted_data.append(col_data)

        # 记录当前df的实际长度
        current_len = min(len(col_data) for col_data in extracted_data)  # 找到最短长度
        lengths_array[i] = current_len

        if current_len < 10:
            print(f"Warning: Segment {i} is too short ({current_len} records), skipping...")
            continue

        # 可视化前两个数据段的特征列
        if i < 2:
            plt.figure(figsize=(12, 6))
            for j, col in enumerate(extract_list):
                plt.subplot(1, len(extract_list), j + 1)
                plt.plot(df[col].values[:min(200, len(df))])  # 只绘制前200个点以便查看
                plt.title(f'Segment {i + 1} - {col}')
                plt.xlabel('Time Index')
                plt.ylabel(col)
            plt.tight_layout()
            plt.show()

        # 填充到max_len长度
        for j, col_data in enumerate(extracted_data):
            padded_array[i, :current_len, j] = col_data[:current_len]

        print(
            f"Processing segment {i + 1}/{n}: original length={current_len}, padded to {max_len}, features={num_features}")

    # 输出最终结果的维度信息
    print(f"\n{'=' * 50}")
    print(f"Final Results:")
    print(f"Padded array shape: {padded_array.shape} (n_samples, max_length, feature_dim)")
    print(f"Lengths array shape: {lengths_array.shape} (n_samples,)")
    print(f"Max sequence length: {max_len}")
    print(f"Total samples: {n}")
    print(f"Extracted columns: {extract_list}")
    print(f"{'=' * 50}")

    return padded_array, lengths_array, cut_data_list


if __name__ == "__main__":
    # 校验文件夹是否存在
    if not os.path.exists(ACTIVE_DIR):
        print(f"❌ Error: Active directory not exist → {ACTIVE_DIR}")
        exit(1)
    if not os.path.exists(CPS_DIR):
        print(f"❌ Error: CPS directory not exist → {CPS_DIR}")
        exit(1)

    padded_array, lengths_array, mapping_list = save_file_for_cluster(extract_list=DATA_COLUMN)

    # 清空输出文件夹（如果存在）
    if os.path.exists(OUTPUT_DIR):
        print(f"Clearing existing output directory: {OUTPUT_DIR}")
        for file in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"  Removed: {file}")

    # 确保输出文件夹存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 保存为.npy文件
    np.save(OUTPUT_DIR + 'data.npy', padded_array)
    np.save(OUTPUT_DIR + 'seq_length.npy', lengths_array)
    print("Arrays saved successfully!")
    print(f"Files saved: data.npy, seq_length.npy")

    # 创建cluster_data目录（如果不存在）
    os.makedirs("cluster_data", exist_ok=True)

    # 保存mapping_list到JSON文件
    with open(OUTPUT_DIR + "data_mapping_list.json", "w", encoding="utf-8") as f:
        json.dump(mapping_list, f, ensure_ascii=False, indent=4)

    print(f"Mapping list saved to {OUTPUT_DIR}/data_mapping_list.json")
    print(f"Total entries in mapping list: {len(mapping_list)}")
