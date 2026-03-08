#!/bin/bash
set -e

echo "=========================================="
echo "Helios 项目依赖安装脚本"
echo "=========================================="

# 检查并激活 conda base 环境
if command -v conda &> /dev/null; then
    echo "检测到 conda，激活 base 环境..."
    eval "$(conda shell.bash hook)"
    conda activate base
    echo "当前环境: $(conda info --envs | grep '*' | awk '{print $1}')"
else
    echo "警告: 未检测到 conda，将在当前 Python 环境安装"
fi

# 设置 HuggingFace 加速
export HF_ENDPOINT=http://hf.x-gpu.com
export HF_HUB_DOWNLOAD_TIMEOUT=180
echo "HuggingFace 加速已设置: $HF_ENDPOINT"

# 安装依赖
echo "安装项目依赖..."
pip install -r requirements.txt

# 清理缓存
echo "清理 Triton 缓存..."
rm -rf ~/.triton/cache/
rm -rf /tmp/torchinductor_*

# 卸载冲突包并重装指定版本
echo "处理版本冲突的包..."
pip uninstall triton torchao xformers wandb tensorflow tensorflow-cpu -y
pip install wandb==0.23.0 triton==3.3.1

# 再次清理缓存
rm -rf ~/.triton/cache/
rm -rf /tmp/torchinductor_*

echo "=========================================="
echo "依赖安装完成！"
echo "=========================================="