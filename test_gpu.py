"""
GPU设备检测脚本

该脚本用于测试TensorFlow和PyTorch是否能够识别当前的GPU设备。
它会输出以下信息：
1. TensorFlow的GPU检测结果
2. PyTorch的GPU检测结果
3. 系统中可用的GPU设备信息
"""

import sys

print("=" * 60)
print("GPU设备检测脚本")
print("=" * 60)

# 测试TensorFlow
print("\n1. 测试TensorFlow:")
try:
    import tensorflow as tf
    print(f"TensorFlow版本: {tf.__version__}")
    
    # 检查GPU是否可用
    gpu_available = tf.test.is_gpu_available()
    print(f"GPU可用: {gpu_available}")
    
    # 列出所有物理设备
    physical_devices = tf.config.list_physical_devices()
    print(f"所有物理设备: {[device.name for device in physical_devices]}")
    
    # 列出GPU设备
    gpu_devices = tf.config.list_physical_devices('GPU')
    print(f"GPU设备数量: {len(gpu_devices)}")
    for i, device in enumerate(gpu_devices):
        print(f"  GPU {i}: {device.name}")
        
    # 检查GPU内存
    if gpu_devices:
        try:
            # 尝试获取GPU内存信息
            for i, gpu in enumerate(gpu_devices):
                with tf.device(f"/GPU:{i}"):
                    # 创建一个简单的操作来测试GPU
                    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
                    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
                    c = tf.multiply(a, b)
                    print(f"  GPU {i} 测试成功: 计算结果 = {c.numpy()}")
        except Exception as e:
            print(f"  GPU测试失败: {e}")
            
except ImportError:
    print("TensorFlow未安装")
except Exception as e:
    print(f"TensorFlow检测错误: {e}")

# 测试PyTorch
print("\n2. 测试PyTorch:")
try:
    import torch
    print(f"PyTorch版本: {torch.__version__}")
    
    # 检查GPU是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {cuda_available}")
    
    if cuda_available:
        # 打印CUDA版本
        print(f"CUDA版本: {torch.version.cuda}")
        
        # 打印GPU数量
        gpu_count = torch.cuda.device_count()
        print(f"GPU设备数量: {gpu_count}")
        
        # 打印每个GPU的信息
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    内存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"    可用内存: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        
        # 测试GPU计算
        try:
            # 创建一个简单的张量并移到GPU
            x = torch.tensor([1.0, 2.0, 3.0])
            x_gpu = x.cuda()
            y_gpu = x_gpu * 2
            print(f"  GPU测试成功: 计算结果 = {y_gpu.cpu().numpy()}")
        except Exception as e:
            print(f"  GPU测试失败: {e}")
    
except ImportError:
    print("PyTorch未安装")
except Exception as e:
    print(f"PyTorch检测错误: {e}")

# 系统信息
print("\n3. 系统信息:")
print(f"Python版本: {sys.version}")
print(f"操作系统: {sys.platform}")

print("\n" + "=" * 60)
print("检测完成")
print("=" * 60)
