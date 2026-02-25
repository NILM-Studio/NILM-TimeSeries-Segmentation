"""
批量特征提取脚本

该脚本专门用于批量处理多个文件的特征提取，支持：
1. 扫描指定文件夹下包含特定字符串的文件
2. 对每个文件执行特征提取
3. 按照指定格式保存提取的特征

使用方法：
    1. 修改脚本中的配置参数
    2. 运行脚本：python batch_feature_extract.py

输出：
    - 提取的特征保存为 .npy 文件
    - 文件名格式：{extract_model}_{xxx}_features_{column_name}_{latent_dim}_dim.npy
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


# ===================== 特征列配置 =====================
# 数据文件中不同特征列的索引映射
# 假设数据文件的第 0 列是时间戳，第 1-4 列是不同的特征
columns_dict = {
    'power': 1,          # 原始功率值
    'cleaned_power': 2,  # 清洗后的功率值
    'high_freq': 3,      # 高频分量
    'low_freq': 4,       # 低频分量
}


def visualize_training_history(training_history, model_name, save_dir, file_name):
    """
    可视化训练历史并保存为JSON文件
    
    Args:
        training_history (dict): 训练历史信息
        model_name (str): 模型名称
        save_dir (str): 保存目录
        file_name (str): 文件名前缀
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
    history_save_dir = os.path.join(save_dir, 'training_history')
    os.makedirs(history_save_dir, exist_ok=True)
    
    # 保存PNG文件
    save_path = os.path.join(history_save_dir, f'{file_name}_{model_name}_training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练历史可视化已保存到: {save_path}")
    
    plt.close()
    
    # 保存训练历史为JSON文件
    json_save_path = os.path.join(history_save_dir, f'{file_name}_{model_name}_training_history.json')
    with open(json_save_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=4, ensure_ascii=False)
    print(f"训练历史JSON文件已保存到: {json_save_path}")


def batch_feature_extract(data_dir, save_dir, target_string, extract_model, column_name, model_config):
    """
    批量执行特征提取
    
    该函数执行以下步骤：
    1. 生成时间戳并创建结果目录
    2. 扫描指定文件夹下包含目标字符串的文件
    3. 对每个符合条件的文件执行特征提取
    4. 按照指定格式保存提取的特征
    5. 可视化训练历史
    
    Args:
        data_dir (str): 数据文件所在目录
        save_dir (str): 特征保存目录
        target_string (str): 文件名中需要包含的目标字符串
        extract_model (str): 特征提取模型类型
        column_name (str): 特征列名称
        model_config (dict): 模型配置参数
    
    Returns:
        list: 提取的特征文件路径列表
    """
    # ===================== 1. 生成时间戳并创建结果目录 =====================
    # 生成时间戳（年月日时分秒）
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"当前时间戳: {timestamp}")
    
    # 创建带时间戳的结果目录
    timestamp_save_dir = os.path.join(save_dir, f'batch_feature_extract_{timestamp}')
    os.makedirs(timestamp_save_dir, exist_ok=True)
    print(f"结果将保存到: {timestamp_save_dir}")
    print("="*70)
    
    # 扫描目录下的文件
    file_list = []
    for file_name in os.listdir(data_dir):
        if target_string in file_name and file_name.endswith('.npy'):
            file_path = os.path.join(data_dir, file_name)
            file_list.append(file_path)
    
    print(f"找到 {len(file_list)} 个包含 '{target_string}' 的文件")
    if not file_list:
        print("未找到符合条件的文件")
        return []
    
    # 用于存储结果文件路径
    result_files = []
    
    # 处理每个文件
    for file_path in file_list:
        # 获取文件名（不含扩展名）
        base_name = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(base_name)[0]
        
        # 提取 xxx 部分（文件名中除了 target_string 外的部分）
        # 这里简单处理，实际可能需要根据具体文件名格式调整
        xxx_part = file_name_without_ext.replace(target_string, '')
        if not xxx_part:
            # 如果替换后为空，使用完整文件名
            xxx_part = file_name_without_ext
        
        print(f"\n处理文件: {base_name}")
        print(f"提取的 xxx 部分: {xxx_part}")
        
        try:
            # 加载数据
            raw_data = np.load(file_path)
            print(f"数据形状: {raw_data.shape}")
            
            # 选择指定的特征列并扩展维度
            data = np.expand_dims(raw_data[:, :, columns_dict[column_name]], axis=-1)
            
            # 提取数据维度信息
            n_samples = data.shape[0]
            timesteps = data.shape[1]
            n_features = data.shape[2]
            
            print(f"数据处理完成 | 总样本数: {n_samples} | 时间步: {timesteps} | 特征数: {n_features}")
            
            # 执行特征提取
            if extract_model == "bilstm_ae":
                print("使用 BiLSTM 自编码器进行特征提取...")
                feature, training_history = bilstm_ae(data, model_config)
                
            elif extract_model == "lstm_ae":
                print("使用 LSTM 自编码器进行特征提取...")
                feature, training_history = lstm_ae(data, model_config)
                
            elif extract_model == "bilstm_ae_attention":
                print("使用 BiLSTM + Attention 自编码器进行特征提取...")
                feature, training_history = bilstm_ae_attention(data, model_config)
                
            else:
                raise ValueError(f"不支持的提取模型: {extract_model}")
            
            # 生成保存文件名
            latent_dim = model_config.get('latent_dim', 64)
            EXTERNAL_TAG = f'_{latent_dim}_dim'
            feature_file_name = f"{extract_model}_{xxx_part}_features_{column_name}{EXTERNAL_TAG}.npy"
            feature_output_path = os.path.join(timestamp_save_dir, feature_file_name)
            
            # 保存特征
            np.save(feature_output_path, feature)
            
            # 可视化训练历史
            visualize_training_history(training_history, extract_model, timestamp_save_dir, file_name_without_ext)
            
            # 打印训练历史摘要
            print(f"训练历史摘要:")
            print(f"  实际训练轮数: {training_history['epochs_trained']}")
            print(f"  最终训练损失: {training_history['loss'][-1]:.6f}")
            print(f"  最终验证损失: {training_history['val_loss'][-1]:.6f}")
            
            print(f"特征提取完成！")
            print(f"特征形状: {feature.shape}")
            print(f"结果已保存到: {feature_output_path}")
            
            result_files.append(feature_output_path)
            
        except Exception as e:
            print(f"处理文件 {base_name} 时出错: {str(e)}")
            continue
    
    print(f"\n批量处理完成！")
    print(f"成功处理 {len(result_files)} 个文件")
    print(f"所有结果已保存到: {timestamp_save_dir}")
    
    return result_files


def main():
    """
    批量特征提取主函数
    
    该函数配置批量处理参数并执行批量特征提取。
    可以根据实际需求修改以下配置参数。
    """
    print("=== 批量特征提取 ===")
    
    # ===================== 批量处理配置 =====================
    # 数据目录：包含待处理的 numpy 文件
    data_dir = "../time_clustering/cluster_data/fridge_test/"
    
    # 保存目录：提取的特征将保存在此目录
    save_dir = "../time_clustering/cluster_data/fridge_test/batch_features/"
    
    # 目标字符串：文件名中需要包含的字符串
    target_string = "data_"
    
    # 特征提取方法
    # 可选值：
    # - "lstm_ae": LSTM 自编码器，输出全局特征 (n_samples, latent_dim)
    # - "bilstm_ae": BiLSTM 自编码器，输出全局特征 (n_samples, latent_dim)
    # - "bilstm_ae_attention": BiLSTM + Attention，输出时间步特征 (n_samples, timesteps, latent_dim)
    extract_model = "lstm_ae"
    
    # 特征列名称
    column_name = "power"  # 可选: 'power', 'cleaned_power', 'high_freq', 'low_freq'
    
    # 模型配置
    model_config = {
        "latent_dim": 64,      # 特征维度
                                # - 对于 lstm_ae 和 bilstm_ae：全局特征维度
                                # - 对于 bilstm_ae_attention：每个时间步的特征维度
        "epochs": 50,         # 训练轮数（最大值，实际可能因早停而提前结束）
        "batch_size": 32,     # 批量大小
        "learning_rate": 0.001,  # 学习率
        "patience": 5,         # 早停耐心值（验证集损失多少个 epoch 没有改善就停止）
    }
    
    # 执行批量特征提取
    result_files = batch_feature_extract(
        data_dir=data_dir,
        save_dir=save_dir,
        target_string=target_string,
        extract_model=extract_model,
        column_name=column_name,
        model_config=model_config
    )
    
    # 打印结果
    print("\n=== 批量处理结果 ===")
    if result_files:
        print("成功生成的特征文件：")
        for file_path in result_files:
            print(f"- {os.path.basename(file_path)}")
    else:
        print("未生成任何特征文件")


if __name__ == "__main__":
    """
    脚本入口点
    
    当直接运行此脚本时，执行批量特征提取流程。
    可以通过修改 main 函数中的配置变量来调整参数。
    """
    main()
