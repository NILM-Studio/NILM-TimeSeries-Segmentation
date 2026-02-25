#!/bin/bash

#SBATCH -J dbscan_washing_machine_seg # 作业名称
#SBATCH -p RTX3090 # 指定分区
#SBATCH --gres=gpu:1 # 申请1个GPU
#SBATCH -c 8 # 申请8个CPU核心
#SBATCH --mem=32G # 申请32GB内存
#SBATCH -o ./slurm_log/job-%x-%j.out # 标准输出日志，%j会被替换为作业ID
#SBATCH -e ./slurm_log/job-%x-%j.err # 标准错误日志
#SBATCH --time=24:00:00 # 增加作业超时时间到12小时


# --- 1. 作业开始，打印基本信息 ---
echo "Job started on: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
# --- 2. 设置软件环境 (参考附录B) ---
module purge
echo "Environment purged."
module load miniconda3/latest cuda-toolkit/12.1
echo "Modules loaded: $(module list -t)"


# --- 3. 激活 Conda 环境 ---
source $(conda info --base)/bin/activate
conda activate nilm_genenal # <--- 修改为你的Conda环境名
echo "Conda environment activated: $CONDA_DEFAULT_ENV"
# --- 4. 执行你的主要命令 ---
echo "Starting Python script..."
python feature_extract.py
# --- 5. 作业结束 ---
echo "Job finished on: $(date)"