#!/usr/bin/env python3
"""
下载 Helios 模型
"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download

# 设置加速
os.environ["HF_ENDPOINT"] = "http://hf.x-gpu.com"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "180"

print("=" * 60)
print("开始下载 Helios-Distilled 模型")
print("=" * 60)

model_id = "BestWishYsh/Helios-Distilled"
local_dir = "./models/Helios-Distilled"

print(f"\n模型仓库: {model_id}")
print(f"本地目录: {local_dir}")
print(f"HF 镜像: {os.environ['HF_ENDPOINT']}")
print("\n开始下载...\n")

try:
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("\n" + "=" * 60)
    print("✅ 模型下载完成！")
    print("=" * 60)
    print(f"\n模型已保存到: {Path(local_dir).absolute()}")

except Exception as e:
    print(f"\n❌ 下载失败: {e}")
    import traceback
    traceback.print_exc()
