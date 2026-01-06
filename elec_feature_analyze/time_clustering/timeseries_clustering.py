import json
import os
import pandas as pd
import numpy as np

# ===================== 1. 配置项（大写常量，统一管理） =====================
ACTIVE_DIR = r"D:\KnowledgeDatabase\ComputerSecience\PROJ_NILM\source_code\DL_mapping\label-studio-test\washing_machine_all\data"
CPS_DIR = r"D:\KnowledgeDatabase\ComputerSecience\PROJ_NILM\source_code\DL_mapping\label-studio-test" \
          r"\washing_machine_all\output"
OUTPUT_DIR = r"./cluster_data/washing_machine_manually/"
APPLIANCE_NAME = "washing_machine"
CSV_ENCODING = "utf-8"  # 若报编码错，可改为"gbk"或"utf-8-sig"
SAVE_NON_MATCH_FILE = True


# ===================== 2. 核心函数：以active文件为核心匹配+读取 =====================
def match_active_with_cps():
    """
    核心逻辑：遍历active文件夹，匹配对应的CPS文件，读取active文件为DataFrame，构建匹配结果
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
            "data": None,  # 存储读取后的DataFrame
            "cps_file": None,  # 对应的CPS文件名（如Changepoint_xxx.csv）
            "cps": None,
            "cut_data": [],
            "match_status": "None",  # 匹配状态：Success/None
        }

        # 检查CPS文件是否存在
        if os.path.exists(cps_target_path):
            current_match["cps_file"] = cps_target_filename
            current_match["match_status"] = "Success"
            try:
                current_match["cps"] = pd.read_csv(cps_target_path, encoding=CSV_ENCODING)
            except Exception as e:
                print(f"Error reading CPS file: {cps_target_path}")
                print(f"Error details: {str(e)}")
                current_match["cps"] = None
        else:
            continue

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
        else:
            if SAVE_NON_MATCH_FILE:
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

    return cut_data_list, match_results


def save_file_for_cluster():
    """
    将切割后的数据展平并填充为相同长度，存储为numpy数组格式以便后续处理
    :returns
    padded_array: 完成展平后的数据，维度为(n, max_len, 1) 或 (n, max_len, 3)
                 当存在cluster_label列时为(n, max_len, 3)，其中：
                 [:, :, 0]为power_new，[:, :, 1]为power，[:, :, 2]为cluster_label
                 否则为(n, max_len, 1)，其中[:, :, 0]为power
    lengths_array: 每个samples的长度，(n, 1)
    """
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

    # 检查是否存在cluster_label列
    has_cluster_label = any('cluster_label' in df.columns for df in ts_list)

    # 第二步：根据是否存在cluster_label列创建不同形状的numpy数组
    n = len(ts_list)
    if has_cluster_label:
        padded_array = np.zeros((n, max_len, 3))  # 3个特征：power, power_new, cluster_label
    else:
        padded_array = np.zeros((n, max_len, 1))  # 1个特征：power

    # 创建用于记录每个df实际数据长度的数组
    lengths_array = np.zeros(n, dtype=int)

    # 第三步：对每个DataFrame进行展平和填充操作
    for i, df in enumerate(ts_list):
        if has_cluster_label:
            # 处理3维数组：power, power_new, cluster_label
            # 提取power列
            power_data = df['power'].values

            # 提取power_new列（假设存在名为power_new或类似名称的列）
            if 'power_new' in df.columns:
                power_new_data = df['power_new'].values
            else:
                power_new_data = power_data  # 如果没有power_new列，复制power列

            # 提取cluster_label列
            if 'cluster_label' in df.columns:
                cluster_label_data = df['cluster_label'].values
            else:
                # 如果没有cluster_label列，用0填充
                cluster_label_data = np.zeros(len(df))

            # 记录当前df的实际长度，剔除长度太短的
            current_len = min(len(power_data), len(power_new_data), len(cluster_label_data))
            lengths_array[i] = current_len
            if current_len < 10:
                continue

            # 填充到max_len长度
            padded_array[i, :current_len, 0] = power_new_data[:current_len]  # power_new
            padded_array[i, :current_len, 1] = power_data[:current_len]  # power
            padded_array[i, :current_len, 2] = cluster_label_data[:current_len]  # cluster_label

        else:
            # 处理1维数组：仅power列
            flat_data = df['power'].values

            # 记录当前df的实际长度
            current_len = len(flat_data)
            lengths_array[i] = current_len

            # 填充到max_len长度
            padded_array[i, :current_len, 0] = flat_data[:current_len]

        print(
            f"Processing segment {i + 1}/{n}: original length={current_len}, padded to {max_len}, has_cluster_label={has_cluster_label}")

    # 输出最终结果的维度信息
    print(f"\n{'=' * 50}")
    print(f"Final Results:")
    print(f"Padded array shape: {padded_array.shape} (n_samples, max_length, feature_dim)")
    print(f"Lengths array shape: {lengths_array.shape} (n_samples,)")
    print(f"Max sequence length: {max_len}")
    print(f"Total samples: {n}")
    print(f"Has cluster_label column: {has_cluster_label}")
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

    padded_array, lengths_array, mapping_list = save_file_for_cluster()

    # 保存为.npy文件
    np.save(OUTPUT_DIR+'data.npy', padded_array)
    np.save(OUTPUT_DIR+'seq_length.npy', lengths_array)
    print("Arrays saved successfully!")
    print(f"Files saved: data.npy, seq_length.npy")

    # 创建cluster_data目录（如果不存在）
    os.makedirs("cluster_data", exist_ok=True)

    # 保存mapping_list到JSON文件
    with open(OUTPUT_DIR + "data_mapping_list.json", "w", encoding="utf-8") as f:
        json.dump(mapping_list, f, ensure_ascii=False, indent=4)

    print(f"Mapping list saved to {OUTPUT_DIR}/data_mapping_list.json")
    print(f"Total entries in mapping list: {len(mapping_list)}")

