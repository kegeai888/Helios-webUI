#!/bin/bash
set -e

echo "=========================================="
echo "Helios 模型下载脚本"
echo "=========================================="

# 设置 HuggingFace 加速
export HF_ENDPOINT=http://hf.x-gpu.com
export HF_HUB_DOWNLOAD_TIMEOUT=180
echo "HuggingFace 加速已设置: $HF_ENDPOINT"

# 检查 huggingface-cli 是否安装
if ! command -v huggingface-cli &> /dev/null; then
    echo "安装 huggingface-cli..."
    pip install "huggingface_hub[cli]"
fi

# 创建 models 目录
mkdir -p models

# 下载模型函数
download_model() {
    local repo_id=$1
    local local_dir=$2

    echo ""
    echo "=========================================="
    echo "下载模型: $repo_id"
    echo "目标目录: $local_dir"
    echo "=========================================="

    if [ -d "$local_dir" ] && [ "$(ls -A $local_dir)" ]; then
        echo "目录 $local_dir 已存在且非空，跳过下载"
        echo "如需重新下载，请先删除该目录: rm -rf $local_dir"
    else
        huggingface-cli download "$repo_id" --local-dir "$local_dir"
        echo "✓ 下载完成: $local_dir"
    fi
}

# 提示用户选择要下载的模型
echo ""
echo "请选择要下载的模型："
echo "1) Helios-Base (基础模型)"
echo "2) Helios-Mid (中等模型)"
echo "3) Helios-Distilled (蒸馏模型，推荐用于 WebUI)"
echo "4) 全部下载"
echo ""
read -p "请输入选项 (1-4): " choice

case $choice in
    1)
        download_model "BestWishYsh/Helios-Base" "./models/Helios-Base"
        ;;
    2)
        download_model "BestWishYsh/Helios-Mid" "./models/Helios-Mid"
        ;;
    3)
        download_model "BestWishYsh/Helios-Distilled" "./models/Helios-Distilled"
        ;;
    4)
        download_model "BestWishYsh/Helios-Base" "./models/Helios-Base"
        download_model "BestWishYsh/Helios-Mid" "./models/Helios-Mid"
        download_model "BestWishYsh/Helios-Distilled" "./models/Helios-Distilled"
        ;;
    *)
        echo "无效选项，退出"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "模型下载完成！"
echo "=========================================="
echo ""
echo "模型已保存到 ./models/ 目录"
echo "现在可以运行 ./start_app.sh 启动 WebUI"
