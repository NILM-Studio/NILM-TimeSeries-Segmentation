import sys

# sys.path.insert(0, "..")

import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set_theme()
# sns.set_color_codes()

from sktime.datasets import load_electric_devices_segmentation
from sktime.detection.clasp import ClaSPSegmentation, find_dominant_window_sizes
from sktime.detection.plotting.utils import (
    plot_time_series_with_change_points,
    plot_time_series_with_profiles,
)

# ts, period_size, true_cps = load_electric_devices_segmentation()
# fig, ax = plot_time_series_with_change_points("Electric Devices", ts, true_cps)
# plt.show()
# print("Done")


# period_size = 100
# clasp = ClaSPSegmentation(period_length=period_size, n_cps=5, fmt="sparse")
# found_cps = clasp.fit_predict(ts)
# profiles = clasp.profiles
# scores = clasp.scores
# print("The found change points are", found_cps.to_numpy())

def clasp_nilm(ts: pd.DataFrame, period_size, n_cps: int = None):
    if n_cps is None:
        clasp = ClaSPSegmentation()
    else:
        clasp = ClaSPSegmentation(period_length=period_size, n_cps=n_cps)
    print("正在运行clasp...")
    found_cps = clasp.fit_predict(ts)
    profiles = clasp.profiles
    scores = clasp.scores
    print("The found change points are", found_cps.to_numpy())
    clasp_visualize(ts, profiles, found_cps, title="Time Series with Change Points")
    print("Done")
    return ts, found_cps


def clasp_visualize(ts: pd.DataFrame, profiles, n_cps, title="Time Series with Change Points"):
    """
    可视化时间序列数据和变化点

    Parameters:
    ts: pd.DataFrame - 时间序列数据，形状为(n,)
    profiles: array-like - 分割评分剖面数组
    n_cps: array-like - 变化点数组
    title: str - 图表标题
    """
    # 创建子图，行数为 1 + len(profiles)（时间序列图 + 每个profile图）
    fig, axes = plt.subplots(len(profiles) + 1, 1, figsize=(12, 4 * (len(profiles) + 1)))

    # 如果只有一个子图，将axes转换为数组以便统一处理
    if len(profiles) == 0:
        axes = [axes]

    # 绘制原始时间序列
    axes[0].plot(ts.index.to_numpy(), ts.values, linewidth=1, color='blue')
    axes[0].set_title(f"{title} - Original Time Series")
    axes[0].set_ylabel('Values')
    axes[0].grid(True, alpha=0.3)

    # 在时间序列图上绘制变化点
    for cp in n_cps['ilocs'].tolist():
        axes[0].axvline(x=cp, color='red', linestyle='-', alpha=0.7, linewidth=1)

    # 遍历并绘制每个profile子图
    for i, profile in enumerate(profiles):
        axes[i + 1].plot(range(len(profile)), profile, linewidth=1, color='green')
        axes[i + 1].set_title(f"Profile {i}")
        axes[i + 1].set_ylabel('Score')
        axes[i + 1].set_xlabel('Split Point')
        axes[i + 1].grid(True, alpha=0.3)

        # 在profile图上也标记变化点
        for cp in n_cps['ilocs'].tolist():
            if cp < len(profile):
                axes[i + 1].axvline(x=cp, color='red', linestyle='-', alpha=0.7, linewidth=1)

    axes[-1].set_xlabel('Time Index')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ts = pd.read_csv(
        r'../../ukdale_disaggregate/active/washing_machine/Washing_Machine_20131216_171003_20131216_174732_305s.csv')
    ts = ts.iloc[:, 1]
    print("数据形状:", ts.shape)
    print("数据类型:", ts.dtype)
    # ts, period_size, true_cps = load_electric_devices_segmentation()
    dominant_period_size = find_dominant_window_sizes(ts)
    print("Dominant Period", dominant_period_size)
    clasp_nilm(ts, dominant_period_size, 3)
