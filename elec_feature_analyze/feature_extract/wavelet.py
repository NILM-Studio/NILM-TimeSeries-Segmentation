import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.colors import LinearSegmentedColormap

def wavelet_decompose_db4(signal_data, wavelet='db4', level=1):
    """
    小波变换：分离高频（细节系数）和低频（近似系数）
    :param signal_data: 1维信号数组
    :param wavelet: 小波基（如'db4'/'haar'/'sym5'）
    :param level: 分解层数（level=1为基础分解）
    :return: cA: 低频近似系数, cD: 高频细节系数
    """
    # 小波分解
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)
    cA = coeffs[0]  # 低频近似系数（核心趋势）
    cD = coeffs[1]  # 高频细节系数（波动/噪声）
    return cA, cD

def wavelet_decompose_bior(signal_data, wavelet='bior3.7', level=2):  # 核心改：bior3.7+level=2
    """
    双正交小波分解，强化低频提纯：
    - bior3.7：频域低通特性最优，适合信号去噪/高频剥离
    - level=2：2级分解，进一步提纯低频
    """
    coeffs = pywt.wavedec(signal_data, wavelet, level=level, mode='symmetric')  # 改：symmetric边界，减少伪影
    cA = coeffs[0]
    # 对低频系数做阈值去噪（剔除残留高频）
    cA_denoised = pywt.threshold(cA, value=np.std(cA)*1.5, mode='soft')  # 软阈值，平滑剔除高频小系数
    return cA_denoised, coeffs[1:]  # 返回去噪后的低频+所有高频细节


def wavelet_scalogram(signal_data, sampling_freq, wavelet='cmor1.5-1.0', scales=None):
    """
    小波尺度图计算（尺度-时间矩阵），并转换为频率-时间矩阵
    :param signal_data: 1维信号数组
    :param sampling_freq: 采样频率（Hz）
    :param wavelet: 连续小波基（cmor系列适合频率分析）
    :param scales: 尺度范围（None则自动生成）
    :return: freq: 频率数组, time: 时间轴, scalogram: 频率-时间幅值矩阵
    """
    # 自动生成尺度范围（覆盖信号主要频率）
    if scales is None:
        # 尺度与频率成反比，这里生成2-64的尺度（对应高频到低频）
        scales = np.arange(2, 64)

    # 连续小波变换（CWT）
    # cwtmatr: 尺度-时间幅值矩阵, frequencies: 对应频率
    cwtmatr, frequencies = pywt.cwt(
        signal_data,
        scales=scales,
        wavelet=wavelet,
        sampling_period=1 / sampling_freq  # 采样周期（秒）
    )

    # 计算时间轴（与输入信号长度匹配）
    time = np.linspace(0, len(signal_data) / sampling_freq, len(signal_data))

    # 幅值取绝对值（能量）
    scalogram = np.abs(cwtmatr)
    return frequencies, time, scalogram


def plot_freq_time_heatmap(frequencies, time, scalogram, title, save_path=None):
    """
    绘制频率-时间热力图
    :param frequencies: 频率数组
    :param time: 时间数组
    :param scalogram: 频率-时间幅值矩阵
    :param title: 图表标题
    :param save_path: 保存路径（None则不保存）
    """
    # 自定义配色（深蓝→红→黄，突出高频能量）
    colors = [(0, 0, 0.8), (0, 0.8, 0.8), (0.8, 0.8, 0), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)

    # 创建画布
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制热力图（pcolormesh适合非均匀网格）
    im = ax.pcolormesh(
        time,
        frequencies,
        scalogram,
        cmap=cmap,
        shading='gouraud'  # 平滑着色
    )

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('小波幅值（能量）', fontsize=12)

    # 设置坐标轴
    ax.set_xlabel('时间 (s)', fontsize=12)
    ax.set_ylabel('频率 (Hz)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # 限制频率范围（聚焦有效频段）
    ax.set_ylim(0, 30)  # 示例：只显示0-30Hz

    # 网格线（增强可读性）
    ax.grid(alpha=0.3, linestyle='--')

    # 保存/显示
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig  # 返回Figure实例，由调用者决定是否show


def main():
    # ===================== 1. 生成/读取数据 =====================
    df = pd.read_csv('../../ukdale_disaggregate/process_data/washing_machine_channel_5'
                     '/Washing_Machine_20121110_182407_20121110_191850_463s.csv')
    sampling_freq = 100
    signal_data = df['power'].values  # 提取信号值
    time_axis = df['timestamp'].values  # 提取时间轴

    # ===================== 2. 第一次小波变换：分离高低频 =====================
    print("执行第一次小波变换（分离高低频）...")
    cA1, cD1 = wavelet_decompose_bior(signal_data, level=1)

    # 补全低频系数长度（与原信号对齐，方便绘图）
    cA1_padded = pywt.upcoef('a', cA1, 'db4', level=1, take=len(signal_data))

    # ===================== 3. 第二次小波变换：分析低频信号 =====================
    print("对低频信号执行二次小波分析（连续小波变换）...")
    # 对低频信号做连续小波变换，得到频率-时间矩阵
    frequencies, time, scalogram = wavelet_scalogram(
        signal_data=cA1_padded,
        sampling_freq=sampling_freq,
        wavelet='cmor1.5-1.0'  # 复Morlet小波，适合频率分析
    )

    # ===================== 4. 绘制频率-时间热力图 =====================
    print("绘制频率-时间热力图...")
    fig = plot_freq_time_heatmap(
        frequencies=frequencies,
        time=time,
        scalogram=scalogram,
        title='低频信号（2Hz趋势）的频率-时间热力图',
        save_path='low_freq_wavelet_heatmap.png'  # 可选保存路径
    )

    # ===================== 5. 辅助绘图：原始信号+高低频分离结果 =====================
    fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    # 原始信号
    ax1.plot(time_axis, signal_data, color='blue', alpha=0.8, label='原始信号')
    ax1.set_title('原始信号（低频+高频+噪声）', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 低频信号（近似系数）
    ax2.plot(time_axis, cA1_padded, color='red', alpha=0.8, label='低频信号（cA1）')
    ax2.set_title('第一次小波变换 - 低频近似信号', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 高频信号（细节系数，补全长度）
    cD1_padded = pywt.upcoef('d', cD1, 'db4', level=1, take=len(signal_data))
    ax3.plot(time_axis, cD1_padded, color='green', alpha=0.8, label='高频信号（cD1）')
    ax3.set_title('第一次小波变换 - 高频细节信号', fontsize=12)
    ax3.set_xlabel('时间 (s)', fontsize=12)
    ax3.legend()
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    fig2.savefig('wavelet_decompose_result.png', dpi=300, bbox_inches='tight')

    # 由调用者决定是否显示图表
    plt.show()


if __name__ == "__main__":
    main()