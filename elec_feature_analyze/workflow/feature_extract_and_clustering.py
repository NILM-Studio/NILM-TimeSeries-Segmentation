"""
特征提取与聚类评估工作流脚本

该脚本打通了整个工作流：
1. 使用特征提取模型提取数据特征
2. 使用 DBSCAN 算法进行聚类评估（eps 扫描模式）

使用方法：
    1. 修改脚本中的配置参数
    2. 运行脚本：python feature_extract_and_clustering.py

输出：
    - 提取的特征文件
    - DBSCAN 聚类评估结果（eps 扫描结果）
    - 可视化图表
"""

import os
import sys
import numpy as np
import tensorflow as tf

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 导入特征提取模块
from feature_extract.feature_extract import run_feature_extract as single_feature_extract
from feature_extract.lstm_ae import lstm_ae
from feature_extract.bilistm_ae import bilstm_ae
from feature_extract.bilstm_ae_attantion import bilstm_ae_attention

# 导入 DBSCAN 模块
sys.path.append(os.path.join(os.path.dirname(__file__), '../time_clustering'))
from dbscan import eps_scan


class FeatureExtractAndClusteringWorkflow:
    """
    特征提取与聚类评估工作流类
    """
    
    def __init__(self, config):
        """
        初始化工作流
        
        Args:
            config (dict): 工作流配置参数
        """
        self.config = config
        self._validate_config()
        self._create_directories()
    
    def _validate_config(self):
        """
        验证配置参数
        """
        required_keys = [
            'data_file', 'result_dir', 'extract_model', 'model_config',
            'dbscan_config', 'appliance_name'
        ]
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"缺少必要配置项: {key}")
    
    def _create_directories(self):
        """
        创建必要的目录
        """
        # 确保结果目录存在
        os.makedirs(self.config['result_dir'], exist_ok=True)
        
        # 创建特征保存目录
        self.feature_dir = os.path.join(self.config['result_dir'], 'features')
        os.makedirs(self.feature_dir, exist_ok=True)
        
        # 创建 DBSCAN 结果目录
        self.dbscan_dir = os.path.join(self.config['result_dir'], 'dbscan_results')
        os.makedirs(self.dbscan_dir, exist_ok=True)
    
    def run_feature_extract(self):
        """
        运行特征提取
        
        Returns:
            str: 提取的特征文件路径
        """
        print("=== 开始特征提取 ===")
        
        # 加载数据
        raw_data = np.load(self.config['data_file'])
        print(f"原始数据形状: {raw_data.shape}")
        
        # 特征列配置
        columns_dict = {
            'power': 1,          # 原始功率值
            'cleaned_power': 2,  # 清洗后的功率值
            'high_freq': 3,      # 高频分量
            'low_freq': 4,       # 低频分量
        }
        
        # 选择特征列
        column_name = self.config.get('column_name', 'cleaned_power')
        data = np.expand_dims(raw_data[:, :, columns_dict[column_name]], axis=-1)
        
        # 提取模型
        extract_model = self.config['extract_model']
        model_config = self.config['model_config']
        
        # 执行特征提取
        if extract_model == "bilstm_ae":
            print("使用 BiLSTM 自编码器进行特征提取...")
            feature = bilstm_ae(data, model_config)
            
        elif extract_model == "lstm_ae":
            print("使用 LSTM 自编码器进行特征提取...")
            feature = lstm_ae(data, model_config)
            
        elif extract_model == "bilstm_ae_attention":
            print("使用 BiLSTM + Attention 自编码器进行特征提取...")
            feature = bilstm_ae_attention(data, model_config)
            
        else:
            raise ValueError(f"不支持的提取模型: {extract_model}")
        
        # 生成特征文件名
        latent_dim = model_config.get('latent_dim', 64)
        feature_file_name = f"{extract_model}_features_{column_name}_{latent_dim}_dim.npy"
        feature_path = os.path.join(self.feature_dir, feature_file_name)
        
        # 保存特征
        np.save(feature_path, feature)
        
        print(f"\n特征提取完成！")
        print(f"特征形状: {feature.shape}")
        print(f"特征保存路径: {feature_path}")
        
        return feature_path
    
    def run_dbscan_eps_scan(self, feature_path):
        """
        运行 DBSCAN eps 扫描
        
        Args:
            feature_path (str): 特征文件路径
        """
        print("\n=== 开始 DBSCAN eps 扫描 ===")
        
        # 加载特征
        feature = np.load(feature_path)
        print(f"加载特征形状: {feature.shape}")
        
        # 如果是时间步特征，需要处理成全局特征
        if len(feature.shape) == 3:
            print("检测到时间步特征，将其平均池化为全局特征...")
            feature = np.mean(feature, axis=1)
            print(f"处理后特征形状: {feature.shape}")
        
        # DBSCAN 配置
        dbscan_config = self.config['dbscan_config']
        
        # 运行 eps 扫描
        print("执行 DBSCAN eps 扫描...")
        
        # 准备 DBSCAN 配置
        scan_config = {
            'min_pts': dbscan_config.get('min_pts', 5),
            'metric': dbscan_config.get('metric', 'euclidean'),
            'normalization_method': dbscan_config.get('normalization_method', 'zscore')
        }
        
        # 构建保存目录
        save_dir = os.path.join(
            self.dbscan_dir, 
            f"eps_scan_{self.config['extract_model']}"
        )
        os.makedirs(save_dir, exist_ok=True)
        
        # 调用 dbscan.py 中的 scan_eps 函数
        from dbscan import scan_eps
        optimal_eps, eps_results = scan_eps(
            data_np=np.load(self.config['data_file']),
            features_matrix=feature,
            config=scan_config,
            save_dir=save_dir,
            eps_start=dbscan_config.get('eps_start', 0.1),
            eps_end=dbscan_config.get('eps_end', 2.0),
            eps_step=dbscan_config.get('eps_step', 0.1)
        )
        
        # 保存扫描结果
        print("保存 DBSCAN 扫描结果...")
        
        # 1. 保存最佳的 eps 以及其相关结果
        optimal_result = next((r for r in eps_results if r['eps'] == optimal_eps), None)
        if optimal_result:
            optimal_eps_data = {
                'optimal_eps': optimal_eps,
                'optimal_result': optimal_result,
                'scan_parameters': {
                    'min_pts': scan_config['min_pts'],
                    'eps_start': dbscan_config.get('eps_start', 0.1),
                    'eps_end': dbscan_config.get('eps_end', 2.0),
                    'eps_step': dbscan_config.get('eps_step', 0.1),
                    'metric': scan_config.get('metric', 'euclidean')
                }
            }
            import json
            with open(os.path.join(save_dir, 'optimal_eps_result.json'), 'w', encoding='utf-8') as f:
                json.dump(optimal_eps_data, f, ensure_ascii=False, indent=2)
            print(f"最佳 EPS 结果已保存到: {os.path.join(save_dir, 'optimal_eps_result.json')}")
        
        # 2. 保存所有 eps 对应的结果
        import json
        with open(os.path.join(save_dir, 'eps_scan_results.json'), 'w', encoding='utf-8') as f:
            json.dump(eps_results, f, ensure_ascii=False, indent=2)
        print(f"所有 EPS 扫描结果已保存到: {os.path.join(save_dir, 'eps_scan_results.json')}")
        
        print("\nDBSCAN eps 扫描完成！")
    
    def run(self):
        """
        运行完整工作流
        """
        print("=== 开始完整工作流 ===")
        print(f"用电器: {self.config['appliance_name']}")
        print(f"特征提取模型: {self.config['extract_model']}")
        
        try:
            # 1. 运行特征提取
            feature_path = self.run_feature_extract()
            
            # 2. 运行 DBSCAN eps 扫描
            self.run_dbscan_eps_scan(feature_path)
            
            print("\n=== 工作流完成 ===")
            print(f"特征文件路径: {feature_path}")
            print(f"DBSCAN 结果目录: {self.dbscan_dir}")
            
        except Exception as e:
            print(f"\n工作流执行失败: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """
    主函数
    
    配置工作流参数并启动执行
    """
    # ===================== 工作流配置 =====================
    config = {
        # 数据配置
        'data_file': '../time_clustering/cluster_data/dishwasher/data_fusion.npy',  # 原始数据文件路径
        'column_name': 'cleaned_power',  # 特征列名称
        'appliance_name': 'dishwasher',  # 用电器名称
        'result_dir': '../workflow_results/',  # 结果保存根目录
        
        # 特征提取配置
        'extract_model': 'bilstm_ae',  # 特征提取模型
        'model_config': {
            'latent_dim': 64,      # 特征维度
            'epochs': 50,         # 训练轮数
            'batch_size': 32,     # 批量大小
            'learning_rate': 0.001,  # 学习率
            'patience': 5,         # 早停耐心值
        },
        
        # DBSCAN 配置
        'dbscan_config': {
            'eps_start': 0.1,      # eps 扫描起始值
            'eps_end': 2.0,        # eps 扫描结束值
            'eps_step': 0.1,       # eps 扫描步长
            'min_pts': 5,          # min_pts 参数
            'scan_name': 'eps_scan',  # 扫描结果目录名称
        }
    }
    
    # 创建并运行工作流
    workflow = FeatureExtractAndClusteringWorkflow(config)
    workflow.run()


if __name__ == "__main__":
    """
    脚本入口点
    """
    main()
