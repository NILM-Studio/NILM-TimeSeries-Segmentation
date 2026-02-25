import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_eps_scan_results(json_path):
    """
    读取 eps 扫描结果 JSON 文件并转换为 DataFrame
    
    Args:
        json_path (str): JSON 文件路径
        
    Returns:
        pd.DataFrame: 转换后的 DataFrame
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # 对 eps 列取两位小数
    df['eps'] = df['eps'].round(2)
    
    return df


def calculate_metrics_statistics(df_list, metrics=None, eps_range=None, labels=None):
    """
    为每个 DataFrame 计算指定指标的统计信息（横切格式）
    
    Args:
        df_list (list): 包含多个 eps 扫描结果的 DataFrame 列表
        metrics (list, optional): 要计算统计信息的指标列表
        eps_range (tuple, optional): eps 范围，格式为 (min_eps, max_eps)
        labels (list, optional): 每个 DataFrame 对应的标签，用于结果标识
        
    Returns:
        dict: 横切格式的统计信息，结构为 {指标: {统计量: {模型标签: 值}}}
              例如: {"sil_score": {"mean": {"LSTM": 0.695, "BiLSTM": 0.578, "BiLSTM+Attention": 0.457}}}
    """
    if metrics is None:
        metrics = ['comprehensive_loss_with_ch', 'comprehensive_loss_without_ch', 'n_clusters']
    
    if labels is None:
        labels = [f'Model {i+1}' for i in range(len(df_list))]
    
    # 确保标签数量与 DataFrame 数量匹配
    if len(labels) != len(df_list):
        labels = [f'Model {i+1}' for i in range(len(df_list))]
    
    # 初始化横切格式的统计信息结构
    statistics = {}
    
    # 为每个指标初始化结构
    for metric in metrics:
        statistics[metric] = {
            'mean': {},
            'std': {},
            'min': {},
            'max': {},
            'median': {},
            'q25': {},
            'q75': {}
        }
    
    # 为每个 DataFrame 计算统计信息并填充到横切结构中
    for i, (df, label) in enumerate(zip(df_list, labels)):
        # 过滤 eps 范围
        if eps_range:
            min_eps, max_eps = eps_range
            df_filtered = df[(df['eps'] >= min_eps) & (df['eps'] <= max_eps)]
        else:
            df_filtered = df
        
        # 为每个指标计算统计信息
        for metric in metrics:
            # 收集指标值
            values = df_filtered[metric].dropna().values
            
            if values.size > 0:
                values_series = pd.Series(values)
                statistics[metric]['mean'][label] = values_series.mean()
                statistics[metric]['std'][label] = values_series.std()
                statistics[metric]['min'][label] = values_series.min()
                statistics[metric]['max'][label] = values_series.max()
                statistics[metric]['median'][label] = values_series.median()
                statistics[metric]['q25'][label] = values_series.quantile(0.25)
                statistics[metric]['q75'][label] = values_series.quantile(0.75)
            else:
                statistics[metric]['mean'][label] = None
                statistics[metric]['std'][label] = None
                statistics[metric]['min'][label] = None
                statistics[metric]['max'][label] = None
                statistics[metric]['median'][label] = None
                statistics[metric]['q25'][label] = None
                statistics[metric]['q75'][label] = None
    
    return statistics


def visualize_specific_metrics(df_list, metrics=None, eps_range=None, figsize=(12, 8), labels=None, appliance_name=None, save_dir=None, extern_tag=''):
    """
    可视化指定的指标，将多个 DataFrame 的数据在同一子图内呈现
    
    Args:
        df_list (list): 包含多个 eps 扫描结果的 DataFrame 列表
        metrics (list, optional): 要可视化的指标列表
        eps_range (tuple, optional): eps 范围，格式为 (min_eps, max_eps)
        figsize (tuple, optional): 图表大小，默认为 (12, 8)
        labels (list, optional): 每个 DataFrame 对应的标签，用于图例
        appliance_name (str, optional): 设备名称，用于图表标题
        save_dir (str, optional): 保存图片的目录
        extern_tag (str, optional): 外部标签，用于图片文件名
    """
    if metrics is None:
        metrics = ['sil_score', 'db_score', 'ch_score', 'n_clusters', 'n_noise']
    
    if labels is None:
        labels = [f'Data {i+1}' for i in range(len(df_list))]
    
    plt.figure(figsize=figsize)
    
    # 添加顶部标题
    if appliance_name:
        plt.suptitle(f'Feature Extraction Comparison - {appliance_name}', fontsize=14, fontweight='bold')
    
    # 创建子图
    n_metrics = len(metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(n_rows, n_cols, i)
        
        # 对每个 DataFrame 绘制折线图
        for j, df in enumerate(df_list):
            # 过滤 eps 范围
            if eps_range:
                min_eps, max_eps = eps_range
                df_filtered = df[(df['eps'] >= min_eps) & (df['eps'] <= max_eps)]
            else:
                df_filtered = df
            
            # 绘制折线图 - 将pandas Series转换为numpy数组以避免索引错误
            plt.plot(df_filtered['eps'].values, df_filtered[metric].values, 'o-', linewidth=2, markersize=4, label=labels[j])
        
        # 设置标题和标签
        plt.title(f'{metric} vs eps')
        plt.xlabel('eps')
        plt.ylabel(metric)
        
        # 添加图例
        plt.legend()
        
        # 添加网格
        plt.grid(alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局以留出标题空间
    
    # 保存图片
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        filename = f'feature_extract_compare_{extern_tag}.png' if extern_tag else 'feature_extract_compare.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    plt.show()


def process_appliance(appliance_name):
    """
    处理单个用电器的数据
    
    Args:
        appliance_name (str): 用电器名称
    """
    print(f"\n{'='*80}")
    print(f"处理用电器: {appliance_name}")
    print(f"{'='*80}")
    
    EXTERN_TAG = ''
    lstm_path = f'./cluster_data/dbscan_result/{appliance_name}/eps_scan_lstm/eps_scan_results.json'
    bilstm_path = f'./cluster_data/dbscan_result/{appliance_name}/eps_scan_bilstm/eps_scan_results.json'
    bilstm_attention_path = f'./cluster_data/dbscan_result/{appliance_name}/eps_scan_bilstm_attention/eps_scan_results.json'
    save_dir = f'./cluster_data/dbscan_result/{appliance_name}/eps_scan_comparison'

    # 读取所有数据并组合为df_list
    print("开始读取数据...")
    df_list = []
    labels = ['LSTM', 'BiLSTM', 'BiLSTM+Attention']
    paths = [lstm_path, bilstm_path, bilstm_attention_path]
    metrics = ['sil_score', 'db_score', 'ch_score', 'n_clusters', 'n_noise']
    
    for i, (path, label) in enumerate(zip(paths, labels)):
        try:
            df = read_eps_scan_results(path)
            df_list.append(df)
            print(f"\n{label} 数据加载完成，前5行数据:")
            print(df.head())
            print(f"数据形状: {df.shape}")
        except Exception as e:
            print(f"\n读取 {label} 数据时出错: {e}")
    
    if not df_list:
        print("\n没有成功加载任何数据，跳过该用电器。")
        return
    
    # 计算统计信息
    print("\n计算统计信息...")
    # 包含所有需要的指标
    metrics_all = ['sil_score', 'db_score', 'ch_score', 'n_clusters', 'n_noise', 'comprehensive_loss_with_ch']
    stats = calculate_metrics_statistics(df_list, metrics=metrics_all, labels=labels)
    
    # 打印横切格式的统计信息
    print("\n横切格式统计信息（按指标分组）:")
    for metric, stat_dict in stats.items():
        print(f"\n{metric}:")
        for stat_type, model_values in stat_dict.items():
            # 跳过count字段
            if stat_type == 'count':
                continue
            print(f"  {stat_type}:")
            for model, value in model_values.items():
                if value is not None:
                    if isinstance(value, float):
                        print(f"    {model}: {value:.6f}")
                    else:
                        print(f"    {model}: {value}")
                else:
                    print(f"    {model}: None")
        print()
    
    # 保存统计数据为JSON文件
    print("\n保存统计数据到JSON文件...")
    import json
    import os
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存文件路径
    json_save_path = os.path.join(save_dir, f'feature_extract_summary.json')
    
    # 保存数据
    with open(json_save_path, 'w', encoding='utf-8') as f:
        # 转换numpy类型为Python原生类型
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            else:
                return obj
        
        # 转换数据类型
        stats_serializable = convert_numpy_types(stats)
        
        # 保存为JSON
        json.dump(stats_serializable, f, ensure_ascii=False, indent=2)
    
    print(f"统计数据已保存到: {json_save_path}")
    
    # 可视化指定 eps 范围的指标
    print("\n可视化指定 eps 范围的指标...")
    visualize_specific_metrics(
        df_list, 
        metrics=metrics, 
        eps_range=(0.0, 2.0), 
        labels=labels,
        appliance_name=appliance_name,
        save_dir=save_dir,
        extern_tag=EXTERN_TAG
    )
    
    print(f"\n用电器 {appliance_name} 处理完成！")
    print(f"{'='*80}")


def main():
    """
    主函数，逐个处理多个用电器的数据
    """
    # 要处理的用电器列表
    appliances = ['dishwasher', 'fridge', 'kettle', 'microwave', 'washing_machine']
    
    # 逐个处理每个用电器
    for appliance in appliances:
        process_appliance(appliance)
    
    print("\n所有用电器处理完成！")


if __name__ == '__main__':
    main()