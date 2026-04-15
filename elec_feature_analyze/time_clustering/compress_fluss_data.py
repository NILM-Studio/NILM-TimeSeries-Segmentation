import os
import zipfile
from pathlib import Path


def find_and_compress_fluss_data(root_dir: str, output_zip: str = "fluss_data_all.zip"):
    """
    遍历指定文件夹下所有名为fluss_data的文件夹，并将其压缩为一个zip文件
    
    Args:
        root_dir: 要搜索的根目录
        output_zip: 输出的zip文件名
    """
    fluss_data_dirs = []
    
    print(f"正在搜索目录: {root_dir}")
    print("-" * 50)
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "fluss_data" in dirnames:
            fluss_data_path = os.path.join(dirpath, "fluss_data")
            fluss_data_dirs.append(fluss_data_path)
            print(f"找到: {fluss_data_path}")
    
    if not fluss_data_dirs:
        print("未找到任何名为'fluss_data'的文件夹!")
        return
    
    print("-" * 50)
    print(f"共找到 {len(fluss_data_dirs)} 个fluss_data文件夹")
    print(f"开始压缩到: {output_zip}")
    print("-" * 50)
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for fluss_dir in fluss_data_dirs:
            fluss_path = Path(fluss_dir)
            parent_name = fluss_path.parent.name
            
            for file_path in fluss_path.rglob('*'):
                if file_path.is_file():
                    arcname = f"{parent_name}/fluss_data/{file_path.name}"
                    zipf.write(file_path, arcname)
                    print(f"  添加: {arcname}")
    
    print("-" * 50)
    zip_size = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"压缩完成! 文件大小: {zip_size:.2f} MB")
    print(f"输出文件: {os.path.abspath(output_zip)}")


if __name__ == "__main__":
    root_directory = input("请输入要搜索的根目录路径(直接回车使用当前目录): ").strip()
    
    if not root_directory:
        root_directory = os.getcwd()
    
    if not os.path.isdir(root_directory):
        print(f"错误: 目录不存在 - {root_directory}")
        exit(1)
    
    find_and_compress_fluss_data(root_directory)
