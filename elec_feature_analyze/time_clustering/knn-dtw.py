import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


# def dbscan_dtw_tslearn(ts_data):
#     """
#     tslearn库的dbscan_dtw方法
#     :return:
#     """
#     # ===================== 2. 数据预处理 =====================
#     ts_dataset = to_time_series_dataset(ts_data)
#     scaler = TimeSeriesScalerMeanVariance()
#     ts_dataset_scaled = scaler.fit_transform(ts_dataset)
#
#     # ===================== 3. 正确调用TimeSeriesDBSCAN =====================
#     dbscan_dtw = TimeSeriesDBSCAN(
#         eps=0.3,  # DTW距离阈值（标准化后）
#         min_samples=3,  # 核心点最小样本数
#         metric="dtw",  # 指定DTW距离
#         n_jobs=-1  # 多线程加速
#     )
#     labels = dbscan_dtw.fit_predict(ts_dataset_scaled)
#
#     # ===================== 4. 结果输出 =====================
#     n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#     n_noise = np.sum(labels == -1)
#
#     print("DBSCAN-DTW 聚类结果（tslearn官方类）：")
#     print(f"聚类数量：{n_clusters}")
#     print(f"噪声点数量：{n_noise}")
#     print("-" * 50)
#     for idx, label in enumerate(labels):
#         if label == -1:
#             print(f"时间序列 {idx}：噪声点")
#         else:
#             print(f"时间序列 {idx}：聚类 {label}")


def dtw_distance(x, y, normalize=True):
    """
    计算两个时间序列之间的DTW距离（增加归一化和对角线限制优化）

    Parameters:
    x, y: array-like, shape (n_samples, )
        时间序列数据
    normalize: bool, default=True
        是否归一化DTW距离（除以序列总长度）

    Returns:
    float: 归一化/非归一化的DTW距离
    """
    x = np.array(x).flatten()
    y = np.array(y).flatten()

    n, m = len(x), len(y)
    max_len = max(n, m)

    # 初始化距离矩阵（增加对角线限制，仅计算带宽内的元素，优化性能）
    window = max(n, m) // 10  # 带宽为序列最大长度的10%
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # 填充距离矩阵（仅在带宽内计算）
    for i in range(1, n + 1):
        # 限制j的范围，仅计算对角线附近的元素
        start_j = max(1, i - window)
        end_j = min(m + 1, i + window)
        for j in range(start_j, end_j):
            cost = abs(x[i - 1] - y[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],  # 插入
                dtw_matrix[i, j - 1],  # 删除
                dtw_matrix[i - 1, j - 1]  # 匹配
            )

    dtw_dist = dtw_matrix[n, m]
    # 归一化：消除序列长度对距离的影响
    if normalize:
        dtw_dist = dtw_dist / max_len

    return dtw_dist


def compute_distance_matrix(time_series_list, normalize=True):
    """
    计算时间序列列表之间的DTW距离矩阵（优化循环效率）

    Parameters:
    time_series_list: list of array-like
        时间序列列表，每个序列形状为(len, )
    normalize: bool, default=True
        是否归一化DTW距离

    Returns:
    np.ndarray: 距离矩阵
    """
    n = len(time_series_list)
    distance_matrix = np.zeros((n, n))

    # 优化：仅计算上三角矩阵，避免重复计算
    for i in range(n):
        x = time_series_list[i]
        for j in range(i + 1, n):
            y = time_series_list[j]
            dist = dtw_distance(x, y, normalize=normalize)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix


class DBSCAN_DTW(BaseEstimator, ClusterMixin):
    """
    基于DTW距离的DBSCAN聚类算法（修复BFS逻辑，增加边界处理）
    """

    def __init__(self, eps=0.5, min_samples=3):
        """
        初始化参数

        Parameters:
        eps: float, default=0.5
            邻域半径（归一化后DTW距离的阈值）
        min_samples: int, default=3
            核心点的最小样本数
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.n_clusters_ = 0
        self.distance_matrix_ = None

    def fit(self, X):
        """
        对时间序列进行DBSCAN聚类（修复BFS逻辑）

        Parameters:
        X: list of array-like
            时间序列列表，每个序列形状为(len, )

        Returns:
        self: object
        """
        # 边界处理：空输入
        if not X:
            self.labels_ = np.array([])
            self.n_clusters_ = 0
            return self

        self.time_series_ = [np.array(ts).flatten() for ts in X]  # 统一格式
        self.n_samples_ = len(self.time_series_)

        # 计算归一化的距离矩阵
        self.distance_matrix_ = compute_distance_matrix(self.time_series_, normalize=True)

        # 初始化标签数组 (-1表示噪声点，-2表示未访问)
        self.labels_ = np.full(self.n_samples_, -2, dtype=int)
        cluster_label = 0

        # 遍历所有样本
        for i in range(self.n_samples_):
            if self.labels_[i] != -2:  # 已访问，跳过
                continue

            # 找到邻域内的点
            neighbors = self._get_neighbors(i)

            # 非核心点：标记为噪声
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1
                continue

            # 核心点：启动BFS，扩展聚类
            self._expand_cluster(i, neighbors, cluster_label)
            cluster_label += 1

        self.n_clusters_ = cluster_label
        return self

    def _get_neighbors(self, point_idx):
        """
        获取指定点eps邻域内的所有点
        """
        neighbors = []
        for i in range(self.n_samples_):
            if self.distance_matrix_[point_idx, i] <= self.eps:
                neighbors.append(i)
        return neighbors

    def _expand_cluster(self, core_idx, neighbors, cluster_label):
        """
        扩展聚类（修复BFS逻辑，过滤已处理点）
        """
        # 初始化队列：仅包含未访问/噪声的邻域点
        queue = [p for p in neighbors if self.labels_[p] in (-1, -2)]
        # 标记核心点为当前聚类
        self.labels_[core_idx] = cluster_label

        while queue:
            current_point = queue.pop(0)  # BFS：队列（FIFO），原代码用pop()是DFS，错误！

            # 未访问的点：标记为当前聚类，然后检查是否是核心点
            if self.labels_[current_point] == -2:
                self.labels_[current_point] = cluster_label
                current_neighbors = self._get_neighbors(current_point)
                # 核心点：将其邻域的未处理点加入队列
                if len(current_neighbors) >= self.min_samples:
                    new_points = [p for p in current_neighbors if self.labels_[p] in (-1, -2)]
                    queue.extend(new_points)
            # 噪声点：归入当前聚类
            elif self.labels_[current_point] == -1:
                self.labels_[current_point] = cluster_label

    def fit_predict(self, X):
        """
        训练模型并返回聚类标签
        :parameter
        X: List对象，成员为(len, 1)的numpy数组
        """
        self.fit(X)
        return self.labels_


def visualize_clusters(time_series_list, labels, title="DBSCAN-DTW Clustering Results"):
    """
    可视化聚类结果（优化鲁棒性和可读性）

    Parameters:
    time_series_list: list of array-like
        时间序列列表
    labels: array-like
        聚类标签
    title: str
        图表标题
    """
    unique_labels = np.unique(labels)
    non_noise_labels = unique_labels[unique_labels != -1]
    n_non_noise = len(non_noise_labels)
    noise_indices = np.where(labels == -1)[0]

    # 处理全噪声场景
    if n_non_noise == 0:
        plt.figure(figsize=(12, 6))
        plt.title(f'{title} - All Noise Points')
        for idx in noise_indices:
            plt.plot(time_series_list[idx].flatten(), alpha=0.5, color='black',
                     label=f'Series {idx}' if idx < 5 else "")
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        return

    # 绘制聚类和噪声点
    plt.figure(figsize=(12, 2 * (n_non_noise + 1)))
    plt.suptitle(title, fontsize=14)

    # 绘制每个聚类
    for idx, cluster_id in enumerate(non_noise_labels):
        plt.subplot(n_non_noise + 1, 1, idx + 1)
        cluster_indices = np.where(labels == cluster_id)[0]
        for seq_idx in cluster_indices:
            plt.plot(time_series_list[seq_idx].flatten(), alpha=0.7, label=f'Series {seq_idx}')
        plt.title(f'Cluster {cluster_id} (n={len(cluster_indices)})')
        plt.ylabel('Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(alpha=0.3)

    # 绘制噪声点
    if len(noise_indices) > 0:
        plt.subplot(n_non_noise + 1, 1, n_non_noise + 1)
        for seq_idx in noise_indices:
            plt.plot(time_series_list[seq_idx].flatten(), alpha=0.7, color='black', label=f'Series {seq_idx}')
        plt.title(f'Noise Points (n={len(noise_indices)})')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# 示例用法
if __name__ == "__main__":
    np.random.seed(42)

    # 生成示例数据：不同长度的时间序列
    time_series_data = []

    # 类别1：正弦波
    for i in range(5):
        length = np.random.randint(50, 100)
        t = np.linspace(0, 4 * np.pi, length)
        series = np.sin(t) + 0.1 * np.random.randn(length)
        time_series_data.append(series)

    # 类别2：余弦波
    for i in range(5):
        length = np.random.randint(50, 100)
        t = np.linspace(0, 4 * np.pi, length)
        series = np.cos(t) + 0.1 * np.random.randn(length)
        time_series_data.append(series)

    # 类别3：线性趋势
    for i in range(3):
        length = np.random.randint(50, 100)
        t = np.linspace(0, 10, length)
        series = t + 0.1 * np.random.randn(length)
        time_series_data.append(series)

    # 添加一些噪声点
    for i in range(2):
        length = np.random.randint(30, 80)
        series = np.random.randn(length) * 0.5
        time_series_data.append(series)

    # 应用DBSCAN-DTW聚类（调整eps为归一化后的值）
    dbscan_dtw = DBSCAN_DTW(eps=0.15, min_samples=3)
    labels = dbscan_dtw.fit_predict(time_series_data)

    # 输出结果
    print("DBSCAN-DTW Clustering Results:")
    print(f"Number of clusters: {dbscan_dtw.n_clusters_}")
    print(f"Number of noise points: {np.sum(labels == -1)}")

    for i, label in enumerate(labels):
        if label == -1:
            print(f"Time Series {i}: Noise Point")
        else:
            print(f"Time Series {i}: Cluster {label}")

    # 可视化结果
    visualize_clusters(time_series_data, labels)
