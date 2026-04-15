#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSV文件可视化工具

功能：
1. 遍历指定文件夹下的所有CSV文件
2. 对指定列进行可视化
3. 支持交互式操作：
   - 直接回车：查看下一个文件
   - 输入数字（空格分隔）：在当前图上添加切分点竖线

使用方法：
    python paper_visual.py <文件夹路径> <列名>
    例如：python paper_visual.py ./data "active power"
    
    带状态过滤：
    python paper_visual.py <文件夹路径> <列名> --status <状态值>
    例如：python paper_visual.py ./data "active power" --status 3
"""

import csv
import os
import sys
import glob
import argparse


def setup_chinese_font():
    """设置中文字体支持"""
    try:
        import matplotlib
        from matplotlib import font_manager
        
        # 尝试找到中文字体
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
        font_found = None
        
        for font_name in chinese_fonts:
            try:
                font_path = font_manager.findfont(font_name, fallback_to_default=False)
                if font_path and font_name.lower() in font_path.lower():
                    font_found = font_name
                    break
            except:
                continue
        
        if font_found:
            import matplotlib.pyplot as plt
            plt.rcParams['font.sans-serif'] = [font_found]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"使用中文字体: {font_found}")
        else:
            print("警告: 未找到中文字体，中文可能显示为方块")
    except Exception as e:
        print(f"字体设置警告: {e}")


def read_csv_column(file_path, column_name):
    """
    读取CSV文件的指定列
    
    参数：
        file_path: CSV文件路径
        column_name: 要读取的列名
    
    返回：
        data: 列数据列表
        column_index: 列索引
    """
    data = []
    column_index = None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            # 读取表头
            header = next(reader)
            
            # 查找列索引
            try:
                column_index = header.index(column_name)
            except ValueError:
                print(f"错误: 列 '{column_name}' 不存在于文件")
                print(f"可用列: {header}")
                return None, None
            
            # 读取数据
            for row in reader:
                if len(row) > column_index:
                    try:
                        value = float(row[column_index])
                        data.append(value)
                    except ValueError:
                        # 跳过非数值数据
                        continue
    except Exception as e:
        print(f"错误: 无法读取文件 {file_path}: {e}")
        return None, None
    
    return data, column_index


def visualize_csv_with_cps(file_path, column_name, cps_points=None):
    """
    可视化CSV文件的指定列，并可选添加切分点竖线
    
    参数：
        file_path: CSV文件路径
        column_name: 要可视化的列名
        cps_points: 切分点列表（索引位置）
    """
    import matplotlib.pyplot as plt
    
    # 读取CSV文件的指定列
    data, column_index = read_csv_column(file_path, column_name)
    
    if data is None:
        return False
    
    # 创建图形
    plt.figure(figsize=(14, 6))
    
    # 绘制数据
    x_indices = list(range(len(data)))
    plt.plot(x_indices, data, linewidth=1.5, color='blue', label=column_name)
    
    # 如果有切分点，添加竖线
    if cps_points:
        for i, cps in enumerate(cps_points):
            # 确保切分点在有效范围内
            if 0 <= cps < len(data):
                plt.axvline(x=cps, color='red', linestyle='--', linewidth=1.5, 
                           label=f'CPS {i+1}' if i == 0 else f'CPS {i+1}')
    
    # 设置标题和标签
    filename = os.path.basename(file_path)
    plt.title(f'{filename} - {column_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Index', fontsize=12)
    plt.ylabel(column_name, fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.show()
    
    return True


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='CSV文件可视化工具')
    parser.add_argument('folder_path', help='CSV文件所在文件夹路径')
    parser.add_argument('column_name', help='要可视化的列名')
    parser.add_argument('--status', type=int, help='只处理文件名最后1个字符等于该状态的CSV文件')
    
    args = parser.parse_args()
    folder_path = args.folder_path
    column_name = args.column_name
    status = args.status
    
    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        sys.exit(1)
    
    # 设置中文字体（在导入matplotlib之后）
    try:
        setup_chinese_font()
    except:
        pass
    
    # 获取所有CSV文件
    csv_pattern = os.path.join(folder_path, "*.csv")
    csv_files = sorted(glob.glob(csv_pattern))
    
    # 根据status参数过滤文件
    if status is not None:
        filtered_files = []
        for file in csv_files:
            filename = os.path.basename(file)
            # 获取文件名的最后1个字符
            last_char = filename[-5]  # 因为.csv是4个字符，所以最后1个有效字符是-5位置
            try:
                file_status = int(last_char)
                if file_status == status:
                    filtered_files.append(file)
            except (ValueError, IndexError):
                # 忽略无法解析状态的文件
                pass
        csv_files = filtered_files
    
    if not csv_files:
        print(f"Error: No CSV files found in folder '{folder_path}'")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV files")
    print(f"Visualizing column: {column_name}")
    print("=" * 60)
    print("Instructions:")
    print("  - Press Enter: View next file")
    print("  - Enter numbers (space-separated): Add CPS vertical lines")
    print("  - Enter 'q', 'quit', or 'e': Exit program")
    print("=" * 60)
    
    # 遍历所有CSV文件
    current_idx = 0
    while current_idx < len(csv_files):
        file_path = csv_files[current_idx]
        filename = os.path.basename(file_path)
        
        print(f"\n[{current_idx + 1}/{len(csv_files)}] Visualizing: {filename}")
        
        # 第一次显示（无切分点）
        cps_points = None
        
        while True:
            # 可视化当前文件
            success = visualize_csv_with_cps(file_path, column_name, cps_points)
            
            if not success:
                break
            
            # 获取用户输入
            try:
                user_input = input(f"\n[{filename}] Enter CPS (space-separated) or press Enter to continue: ").strip()
            except KeyboardInterrupt:
                print("\nExiting program")
                sys.exit(0)
            
            # 处理用户输入
            if user_input.lower() in ['q', 'quit', 'exit', 'e']:
                print("Exiting program")
                sys.exit(0)
            
            if user_input == "":
                # 直接回车，查看下一个文件
                break
            
            # 尝试解析切分点
            try:
                # 解析空格分隔的数字
                cps_points = [int(x) for x in user_input.split()]
                print(f"Added CPS points: {cps_points}")
                # 继续循环，重新显示图形（带切分点）
            except ValueError:
                print("Error: Please enter valid integers (space-separated)")
                cps_points = None
        
        current_idx += 1
    
    print("\nAll files processed!")


if __name__ == "__main__":
    main()
