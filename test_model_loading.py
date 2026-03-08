#!/usr/bin/env python3
"""
测试 Helios 模型加载
"""
import sys
import torch
from pathlib import Path

def test_model_loading():
    """测试模型加载是否成功"""

    print("=" * 60)
    print("Helios 模型加载测试")
    print("=" * 60)

    # 检查 CUDA
    print(f"\n1. CUDA 可用性检查")
    print(f"   CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA 版本: {torch.version.cuda}")
        print(f"   GPU 数量: {torch.cuda.device_count()}")
        print(f"   当前 GPU: {torch.cuda.get_device_name(0)}")

    # 检查模型目录
    model_path = Path("./models/Helios-Distilled")
    print(f"\n2. 模型目录检查")
    print(f"   路径: {model_path.absolute()}")
    print(f"   存在: {model_path.exists()}")

    if not model_path.exists():
        print("\n❌ 错误: 模型目录不存在")
        print("   请先运行: bash download_models.sh")
        return False

    # 列出模型文件
    files = list(model_path.rglob("*"))
    print(f"   文件数量: {len(files)}")

    # 检查关键文件
    key_files = ["config.json", "model_index.json"]
    print(f"\n3. 关键文件检查")
    for key_file in key_files:
        exists = (model_path / key_file).exists()
        status = "✓" if exists else "✗"
        print(f"   {status} {key_file}")

    # 尝试加载模型组件
    print(f"\n4. 模型组件加载测试")

    try:
        from diffusers import AutoencoderKLWan, HeliosDMDScheduler, HeliosPyramidPipeline

        print("   [1/3] 加载 VAE...")
        vae = AutoencoderKLWan.from_pretrained(
            str(model_path),
            subfolder="vae",
            torch_dtype=torch.float32
        )
        print("   ✓ VAE 加载成功")

        print("   [2/3] 加载 Scheduler...")
        scheduler = HeliosDMDScheduler.from_pretrained(
            str(model_path),
            subfolder="scheduler"
        )
        print("   ✓ Scheduler 加载成功")

        print("   [3/3] 加载 Pipeline...")
        pipe = HeliosPyramidPipeline.from_pretrained(
            str(model_path),
            vae=vae,
            scheduler=scheduler,
            torch_dtype=torch.bfloat16,
            is_distilled=True
        )
        print("   ✓ Pipeline 加载成功")

        # 检查 pipeline 组件
        print(f"\n5. Pipeline 组件检查")
        print(f"   ✓ Transformer: {type(pipe.transformer).__name__}")
        print(f"   ✓ VAE: {type(pipe.vae).__name__}")
        print(f"   ✓ Scheduler: {type(pipe.scheduler).__name__}")
        print(f"   ✓ Text Encoder: {type(pipe.text_encoder).__name__}")

        # 尝试移动到 CUDA（如果可用）
        if torch.cuda.is_available():
            print(f"\n6. CUDA 加载测试")
            print("   正在将模型移动到 GPU...")
            pipe.to("cuda")
            print("   ✓ 模型已成功移动到 GPU")

            # 设置注意力后端
            try:
                pipe.transformer.set_attention_backend("_flash_3_hub")
                print("   ✓ Flash Attention 3 已启用")
            except Exception:
                try:
                    pipe.transformer.set_attention_backend("flash_hub")
                    print("   ✓ Flash Attention 已启用")
                except Exception as e:
                    print(f"   ⚠ Flash Attention 不可用: {e}")

        print("\n" + "=" * 60)
        print("✅ 所有测试通过！模型加载成功")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ 模型加载失败")
        print(f"   错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
