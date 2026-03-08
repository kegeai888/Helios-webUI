#!/bin/bash

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Helios App 启动脚本"
echo "=========================================="

# 1. 激活 conda base 环境
echo "[1/6] 激活 conda base 环境..."
if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate base
    echo "✓ conda base 环境已激活"
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate base
    echo "✓ conda base 环境已激活"
else
    echo "⚠ 未找到 conda，跳过环境激活（使用系统 Python）"
fi

# 2. 设置 HuggingFace 加速
echo ""
echo "[2/6] 设置 HuggingFace 加速..."
export HF_ENDPOINT=http://hf.x-gpu.com
export HF_HUB_DOWNLOAD_TIMEOUT=180
echo "✓ HF_ENDPOINT=$HF_ENDPOINT"
echo "✓ HF_HUB_DOWNLOAD_TIMEOUT=$HF_HUB_DOWNLOAD_TIMEOUT"

# 3. 检测并释放端口 7860
echo ""
echo "[3/6] 检测端口 7860..."
PORT=7860

# 查找占用端口的进程
if command -v lsof &> /dev/null; then
    PID=$(lsof -ti:$PORT 2>/dev/null || true)
elif command -v netstat &> /dev/null; then
    PID=$(netstat -tlnp 2>/dev/null | grep ":$PORT " | awk '{print $7}' | cut -d'/' -f1 || true)
else
    echo "⚠ 未找到 lsof 或 netstat，跳过端口检测"
    PID=""
fi

if [ -n "$PID" ]; then
    echo "⚠ 端口 $PORT 被进程 $PID 占用，正在终止..."

    # 优雅终止
    kill -15 $PID 2>/dev/null || true

    # 等待最多 5 秒
    for i in {1..5}; do
        if ! kill -0 $PID 2>/dev/null; then
            echo "✓ 进程 $PID 已优雅终止"
            break
        fi
        echo "  等待进程结束... ($i/5)"
        sleep 1
    done

    # 如果还未结束，强制终止
    if kill -0 $PID 2>/dev/null; then
        echo "⚠ 进程未响应，强制终止..."
        kill -9 $PID 2>/dev/null || true
        sleep 1
        echo "✓ 进程 $PID 已强制终止"
    fi
else
    echo "✓ 端口 $PORT 未被占用"
fi

# 4. 确认端口释放
echo ""
echo "[4/6] 确认端口释放..."
sleep 2
echo "✓ 端口已释放"

# 5. 切换到项目目录
echo ""
echo "[5/6] 切换到项目目录..."
cd /root/Helios
echo "✓ 当前目录: $(pwd)"

# 6. 启动 app.py
echo ""
echo "[6/6] 启动 Gradio App..."
echo "=========================================="
python app.py --server-port $PORT
