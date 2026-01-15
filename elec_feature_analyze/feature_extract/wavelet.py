import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class NILMWaveletAnalyzer:
    def __init__(self, wavelet='db4', level=5):
        self.wavelet = wavelet
        self.level = level

    def analyze_signal(self, signal):
        """
        对NILM信号进行小波分析
        """
        # 执行离散小波变换
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)

        # 提取特征
        features = self.extract_features(coeffs)

        # 重构信号用于可视化
        reconstructed = pywt.waverec(coeffs, self.wavelet)

        return coeffs, features, reconstructed

    def extract_features(self, coeffs):
        """
        从小波系数中提取特征
        """
        features = []

        # 近似系数统计特征
        features.extend([
            np.mean(coeffs[0]),
            np.std(coeffs[0]),
            np.max(coeffs[0]),
            np.min(coeffs[0])
        ])

        # 详细系数统计特征
        for detail_coeff in coeffs[1:]:
            features.extend([
                np.mean(np.abs(detail_coeff)),
                np.std(detail_coeff),
                np.max(np.abs(detail_coeff)),
                np.sum(detail_coeff ** 2)  # 能量
            ])

        return np.array(features)

    def detect_appliance_events(self, signal, threshold=2.0):
        """
        检测电器事件
        """
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)

        # 使用最高频段详细系数检测事件
        detail_coeffs = np.abs(coeffs[-1])
        mean_detail = np.mean(detail_coeffs)

        # 简单阈值检测
        events = np.where(detail_coeffs > threshold * mean_detail)[0]

        return events

    def visualize_wavelet_result(self, coeffs):
        # 创建子图
        fig, axes = plt.subplots(self.level + 1, 1, figsize=(12, 2 * (self.level + 1)))
        fig.suptitle('DWT Coefficients')

        # 绘制近似系数（低频部分）
        axes[0].plot(coeffs[0])
        axes[0].set_title('Approximation Coefficients (A{})'.format(self.level))
        axes[0].grid(True)

        # 绘制详细系数（高频部分）
        for i in range(1, self.level + 1):
            axes[i].plot(coeffs[i])
            axes[i].set_title('Detail Coefficients (D{})'.format(self.level - i + 1))
            axes[i].grid(True)

        plt.tight_layout()
        plt.show()

        return coeffs

    def plot_wavelet_coefficients_heatmap(self, coeffs):
        """
        绘制小波系数热力图
        """
        # 创建系数矩阵
        max_len = max(len(c) for c in coeffs)
        coeff_matrix = np.zeros((len(coeffs), max_len))

        for i, coeff in enumerate(coeffs):
            coeff_matrix[i, :len(coeff)] = coeff

        plt.figure(figsize=(12, 6))
        plt.imshow(coeff_matrix, aspect='auto', cmap='viridis')
        plt.colorbar(label='Coefficient Value')
        plt.xlabel('Time/Sample Index')
        plt.ylabel('Scale/Level')
        plt.title('Wavelet Coefficients Heatmap')
        plt.yticks(range(len(coeffs)),
                   ['A{}'.format(len(coeffs) - 1)] +
                   ['D{}'.format(len(coeffs) - i) for i in range(1, len(coeffs))])
        plt.show()


# 使用示例
# analyzer = NILMWaveletAnalyzer(wavelet='db4', level=5)
# coeffs, features, reconstructed = analyzer.analyze_signal(active_power_data)
# events = analyzer.detect_appliance_events(active_power_data)

# 加载数据 (需要替换为实际文件路径)
df = pd.read_csv('../../process_dataset/Air-condition/Air_condition.csv')[:6000]
power_data = df.iloc[:, 5].values

# 创建分析器
analyzer = NILMWaveletAnalyzer(wavelet='db4', level=5)
# 执行分析
coeffs, features, reconstructed = analyzer.analyze_signal(power_data)
# 可视化结果
analyzer.visualize_wavelet_result(coeffs)
analyzer.plot_wavelet_coefficients_heatmap(coeffs)
# 事件检测
events = analyzer.detect_appliance_events(power_data, threshold=2.5)
print(f"检测到电器事件数量: {len(events)}")
# 特征分析
print(f"提取特征数量: {len(features)}")
