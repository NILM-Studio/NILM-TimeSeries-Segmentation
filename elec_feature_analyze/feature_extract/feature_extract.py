"""
统一特征提取脚本

该脚本提供了一个统一的接口来调用三种不同的特征提取方法：
1. LSTM 自编码器 (lstm_ae): 提取全局潜空间特征
2. BiLSTM 自编码器 (bilstm_ae): 提取全局潜空间特征（双向）
3. BiLSTM + Attention 自编码器 (bilstm_ae_attention): 提取时间步级别的特征

使用方法：
    1. 修改 extract_model 变量选择特征提取方法
    2. 修改 model_config 字典配置模型参数
    3. 修改数据路径和特征列名称
    4. 运行脚本：python feature_extract.py

输出：
    - 提取的特征保存为 .npy 文件
    - 文件名格式：{extract_model}_features_{column_name}_{latent_dim}_dim.npy
"""

import os
import json
import datetime
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from lstm_ae import lstm_ae
from bilistm_ae import bilstm_ae
from bilstm_ae_attantion import bilstm_ae_attention


# ===================== 数据路径配置 =====================
# 数据文件路径：包含时序数据的 numpy 数组文件
# 数据形状应为 (n_samples, timesteps, n_features)
appliance_name = "washing_machine"  # 可选: "washing_machine", "dishwasher", "fridge"
data_file = f"../time_clustering/cluster_data/{appliance_name}/data_fusion.npy"

# 序列长度文件路径：包含每个样本的真实长度（用于不等长数据处理）
seq_file = f"../time_clustering/cluster_data/{appliance_name}/seq_length_fusion.npy"

# 结果保存目录：提取的特征将保存在此目录
result_dir = f"../time_clustering/cluster_data/{appliance_name}/"


# ===================== 特征提取方法选择 =====================
# 可选值：
# - "lstm_ae": LSTM 自编码器，输出全局特征 (n_samples, latent_dim)
# - "bilstm_ae": BiLSTM 自编码器，输出全局特征 (n_samples, latent_dim)
# - "bilstm_ae_attention": BiLSTM + Attention，输出时间步特征 (n_samples, timesteps, latent_dim)
extract_model = "lstm_ae"


# ===================== 模型配置参数 =====================
model_config = {
  "latent_dim": 64,      # 特征维度
                        # - 对于 lstm_ae 和 bilstm_ae：全局特征维度
                        # - 对于 bilstm_ae_attention：每个时间步的特征维度
  "epochs": 50,         # 训练轮数（最大值，实际可能因早停而提前结束）
  "batch_size": 32,     # 批量大小
  "learning_rate": 0.001,  # 学习率
  "patience": 5,         # 早停耐心值（验证集损失多少个 epoch 没有改善就停止）
}


# ===================== 特征列配置 =====================
# 数据文件中不同特征列的索引映射
# 假设数据文件的第 0 列是时间戳，第 1-4 列是不同的特征
columns_dict = {
    'power': 1,          # 原始功率值
    'cleaned_power': 2,  # 清洗后的功率值
    'high_freq': 3,      # 高频分量
    'low_freq': 4,       # 低频分量
}

# 选择要提取的特征列
column_name = 'cleaned_power'  # 可选: 'power', 'cleaned_power', 'high_freq', 'low_freq'


# ===================== 输出文件名配置 =====================
# 外部标签：用于区分不同配置的特征文件
EXTERNAL_TAG = f'_{model_config["latent_dim"]}_dim'

# 特征文件名：格式为 {extract_model}_features_{column_name}{EXTERNAL_TAG}.npy
# 例如：bilstm_ae_features_cleaned_power_64_dim.npy
feature_file_name = f"{extract_model}_features_{column_name}{EXTERNAL_TAG}.npy"


def visualize_training_history(training_history, model_name, result_dir):
    """
    可视化训练历史并保存为JSON文件
    
    Args:
        training_history (dict): 训练历史信息
        model_name (str): 模型名称
        result_dir (str): 结果保存目录
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制训练损失和验证损失
    plt.plot(training_history['loss'], label='Training Loss')
    plt.plot(training_history['val_loss'], label='Validation Loss')
    
    plt.title(f'Training History - {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    # 保存可视化结果
    save_dir = os.path.join(result_dir, 'training_history')
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存PNG文件
    save_path = os.path.join(save_dir, f'{model_name}_training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练历史可视化已保存到: {save_path}")
    
    plt.close()
    
    # 保存训练历史为JSON文件
    json_save_path = os.path.join(save_dir, f'{model_name}_training_history.json')
    with open(json_save_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=4, ensure_ascii=False)
    print(f"训练历史JSON文件已保存到: {json_save_path}")


def run_feature_extract():
    """
    执行特征提取的主函数
    
    该函数执行以下步骤：
    1. 生成时间戳并创建结果目录
    2. 打印 TensorFlow 和 GPU 信息
    3. 加载和预处理数据
    4. 根据选择的模型提取特征
    5. 保存提取的特征到文件
    6. 可视化训练历史
    
    Returns:
        dict: 包含特征提取结果和训练历史的字典
    
    Raises:
        ValueError: 当 extract_model 不是支持的模型时
    """
    # ===================== 1. 生成时间戳并创建结果目录 =====================
    # 生成时间戳（年月日时分秒）
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"当前时间戳: {timestamp}")
    
    # 创建带时间戳的结果目录
    timestamp_result_dir = os.path.join(result_dir, f'feature_extract_{timestamp}')
    os.makedirs(timestamp_result_dir, exist_ok=True)
    print(f"结果将保存到: {timestamp_result_dir}")
    print("="*70)

    # ===================== 2. 打印 TensorFlow 和 GPU 信息 =====================
    print("TensorFlow版本：", tf.__version__)
    print("="*50)
    print("是否检测到GPU：", tf.config.list_physical_devices('GPU'))
    print("="*50)
    print("所有可用计算设备：", tf.config.list_physical_devices())
    gpu_devices = tf.config.list_logical_devices('GPU')
    print("GPU逻辑设备详情：", gpu_devices)

    # ===================== 3. 加载和预处理数据 =====================
    # 加载原始数据文件（包含所有特征）
    raw_data = np.load(data_file)
    print(f"原始数据形状: {raw_data.shape}")
    
    # 选择指定的特征列并扩展维度
    # raw_data[:, :, columns_dict[column_name]]: 选择特定列，形状为 (n_samples, timesteps)
    # np.expand_dims(..., axis=-1): 在最后添加一个维度，形状变为 (n_samples, timesteps, 1)
    data = np.expand_dims(raw_data[:, :, columns_dict[column_name]], axis=-1)
    
    # 加载序列长度文件
    seq_len = np.load(seq_file).astype(np.int32)  # 明确指定整型

    # 提取数据维度信息
    n_samples = data.shape[0]      # 样本数量
    timesteps = data.shape[1]       # 时间步长度（填充后的统一长度）
    n_features = data.shape[2]      # 特征数量（单特征时为1）

    # 打印数据信息
    print("="*50)
    print(f"对{appliance_name}数据进行特征提取")
    print(f"数据加载完成 | 总样本数: {n_samples} | 填充后时间步: {timesteps} | 输入特征数: {n_features}")
    print(f"样本真实长度范围: {np.min(seq_len)} ~ {np.max(seq_len)}")
    print(f"选择的特征索引: {column_name} | 输入数据最终形状: {data.shape}")

    # ===================== 3. 根据选择的模型提取特征 =====================
    if extract_model == "bilstm_ae":
        print("\n使用 BiLSTM 自编码器进行特征提取...")
        print("输出特征形状: (n_samples, latent_dim) - 全局潜空间特征")
        feature, training_history = bilstm_ae(data, model_config)
        
    elif extract_model == "lstm_ae":
        print("\n使用 LSTM 自编码器进行特征提取...")
        print("输出特征形状: (n_samples, latent_dim) - 全局潜空间特征")
        feature, training_history = lstm_ae(data, model_config)
        
    elif extract_model == "bilstm_ae_attention":
        print("\n使用 BiLSTM + Attention 自编码器进行特征提取...")
        print("输出特征形状: (n_samples, timesteps, latent_dim) - 时间步级别特征")
        feature, training_history = bilstm_ae_attention(data, model_config)
        
    else:
        raise ValueError(f"不支持的提取模型: {extract_model}。"
                        f"可选值: 'lstm_ae', 'bilstm_ae', 'bilstm_ae_attention'")

    # ===================== 5. 保存提取的特征 =====================
    # 构建完整的输出路径
    feature_output_path = os.path.join(timestamp_result_dir, feature_file_name)
    
    # 保存特征到 numpy 文件
    np.save(feature_output_path, feature)
    
    print(f"\n特征提取完成！")
    print(f"特征形状: {feature.shape}")
    print(f"结果已保存到: {feature_output_path}")
    
    # ===================== 6. 可视化训练历史 =====================
    print("\n可视化训练历史...")
    visualize_training_history(training_history, extract_model, timestamp_result_dir)
    
    # 构建结果字典
    result = {
        'feature_path': feature_output_path,
        'feature_shape': feature.shape,
        'training_history': training_history,
        'model_name': extract_model,
        'timestamp': timestamp,
        'result_dir': timestamp_result_dir
    }
    
    return result


if __name__ == "__main__":
    """
    脚本入口点
    
    当直接运行此脚本时，执行特征提取流程。
    可以通过修改脚本顶部的配置变量来调整参数。
    """
    result = run_feature_extract()
    
    # 打印训练历史摘要
    print("\n=== 训练历史摘要 ===")
    print(f"模型名称: {result['model_name']}")
    print(f"实际训练轮数: {result['training_history']['epochs_trained']}")
    print(f"最终训练损失: {result['training_history']['loss'][-1]:.6f}")
    print(f"最终验证损失: {result['training_history']['val_loss'][-1]:.6f}")
    
    # 计算损失改进率
    initial_loss = result['training_history']['loss'][0]
    final_loss = result['training_history']['loss'][-1]
    improvement_rate = ((initial_loss - final_loss) / initial_loss) * 100
    print(f"损失改进率: {improvement_rate:.2f}%")
    
    print("\n=== 工作流完成 ===")
    print(f"特征文件路径: {result['feature_path']}")
    print(f"特征形状: {result['feature_shape']}")
