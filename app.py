import os
from datetime import datetime
from pathlib import Path
import time

import gradio as gr
import spaces
import torch

from diffusers import AutoencoderKLWan, HeliosDMDScheduler, HeliosPyramidPipeline
from diffusers.utils import export_to_video, load_image, load_video


# ---------------------------------------------------------------------------
# 直接加载模型到GPU（48GB显存模式）
# ---------------------------------------------------------------------------
MODEL_ID = "./models/Helios-Distilled"

print("=" * 60)
print("正在加载 Helios 模型到 GPU...")
print("=" * 60)

vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
scheduler = HeliosDMDScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
pipe = HeliosPyramidPipeline.from_pretrained(
    MODEL_ID, vae=vae, scheduler=scheduler, torch_dtype=torch.bfloat16, is_distilled=True
)

print("正在将模型移动到 GPU...")
pipe.to("cuda")

try:
    pipe.transformer.set_attention_backend("_flash_3_hub")
    print("✓ Flash Attention 3 已启用")
except Exception:
    try:
        pipe.transformer.set_attention_backend("flash_hub")
        print("✓ Flash Attention 已启用")
    except Exception:
        print("⚠ Flash Attention 不可用，使用默认注意力机制")

print("=" * 60)
print("✅ 模型已加载到 GPU，准备就绪")
print("=" * 60)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@spaces.GPU(duration=300)
def generate_video(
    mode: str,
    prompt: str,
    image_input,
    video_input,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    seed: int,
    is_amplify_first_chunk: bool,
    progress=gr.Progress(track_tqdm=True),
):
    if not prompt:
        raise gr.Error("请提供提示词")

    # 处理随机种子：-1表示随机生成
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()

    generator = torch.Generator(device="cuda").manual_seed(int(seed))

    kwargs = {
        "prompt": prompt,
        "height": int(height),
        "width": int(width),
        "num_frames": int(num_frames),
        "guidance_scale": 1.0,
        "generator": generator,
        "output_type": "np",
        "pyramid_num_inference_steps_list": [
            int(num_inference_steps),
            int(num_inference_steps),
            int(num_inference_steps),
        ],
        "is_amplify_first_chunk": is_amplify_first_chunk,
    }

    if mode == "图片生成视频" and image_input is not None:
        img = load_image(image_input).resize((int(width), int(height)))
        kwargs["image"] = img
    elif mode == "视频生成视频" and video_input is not None:
        kwargs["video"] = load_video(video_input)

    t0 = time.time()
    output = pipe(**kwargs).frames[0]
    elapsed = time.time()- t0

    # 保存到 outputs 目录，文件名带时间戳
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = Path("./outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"outputs_{timestamp}.mp4"

    export_to_video(output, str(output_path), fps=24)
    info = f"生成耗时 {elapsed:.1f}秒 · {num_frames} 帧 · {height}×{width} · 种子:{seed}"
    return str(output_path), info

# ---------------------------------------------------------------------------
# UI Setup
# ---------------------------------------------------------------------------
def update_conditional_visibility(mode):
    if mode == "图片生成视频":
        return gr.update(visible=True), gr.update(visible=False)
    elif mode == "视频生成视频":
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False)

def update_resolution(resolution_str):
    """从分辨率字符串解析宽度和高度"""
    width, height = map(int, resolution_str.split('x'))
    return height, width


CSS = """
/* 全局样式 - 适配桌面端浏览器 */
body {
    margin: 0;
    padding: 0;
    overflow-x: hidden;
}

/* Gradio 容器 - 宽度90%，左右各留5% */
.gradio-container {
    max-width: 90% !important;
    width: 90% !important;
    margin: 0 auto !important;
    padding: 0 !important;
}

/* 移除 Gradio 默认的最大宽度限制 */
.contain {
    max-width: none !important;
}

/* 标题横幅 - 全宽 */
#header-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    text-align: center;
    padding: 40px 20px;
    margin: -20px -20px 30px -20px;
    border-radius: 0;
    width: calc(100% + 40px);
    margin-left: -20px;
    margin-right: -20px;
}

#header-banner h1 {
    color: white;
    font-size: 2.8em;
    margin: 0 0 20px 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    font-weight: 600;
}

#header-banner .subtitle {
    color: white;
    font-size: 1.2em;
    line-height: 1.8;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
    margin: 10px 0;
}

/* 主内容区域 */
.main-content {
    padding: 30px 0;
    width: 100%;
}

/* 确保所有行和列占满宽度 */
.gradio-row {
    width: 100% !important;
    gap: 30px;
}

.gradio-column {
    min-width: 0 !important;
}

/* 输入组件样式优化 */
.gradio-textbox, .gradio-number, .gradio-slider, .gradio-radio, .gradio-checkbox {
    width: 100% !important;
}

/* 按钮样式 */
.gradio-button {
    width: 100% !important;
    padding: 15px !important;
    font-size: 1.1em !important;
}

/* 视频输出区域 */
.gradio-video {
    width: 100% !important;
}
"""

with gr.Blocks(title="Helios 视频生成") as demo:
    gr.HTML(
        """
        <div id="header-banner">
            <h1>🎬 Helios 长视频生成系统</h1>
            <div class="subtitle">
                webUI二次开发 by 科哥 | 微信：312088415 公众号：科哥玩AI<br>
                承诺永远开源使用 但是需要保留本人版权信息！
            </div>
        </div>
        """
    )

    with gr.Row(elem_classes="main-content"):
        with gr.Column(scale=1):
            mode = gr.Radio(
                choices=["文本生成视频", "图片生成视频", "视频生成视频"],
                value="文本生成视频",
                label="生成模式",
            )
            image_input = gr.Image(label="输入图片（图片生成视频模式）", type="filepath", visible=False)
            video_input = gr.Video(label="输入视频（视频生成视频模式）", visible=False)
            prompt = gr.Textbox(
                label="提示词",
                lines=4,
                placeholder="请输入视频描述...",
                value="一条色彩鲜艳的热带鱼在清澈的海洋中优雅地游动，周围是五颜六色的珊瑚礁。",
            )
            with gr.Accordion("高级设置", open=False):
                resolution = gr.Dropdown(
                    choices=[
                        "640x384",    # 默认 (5:3)
                        "512x512",    # 正方形
                        "640x360",    # 360p (16:9)
                        "720x480",    # SD标清 (3:2)
                        "768x768",    # 正方形高清
                        "1024x576",   # 16:9宽屏
                        "1280x720",   # HD 720p (16:9)
                        "1920x1080",  # Full HD 1080p (16:9)
                    ],
                    value="640x384",
                    label="分辨率 (宽×高)",
                )
                with gr.Row():
                    height = gr.Number(value=384, label="高度", precision=0, visible=False)
                    width = gr.Number(value=640, label="宽度", precision=0, visible=False)
                with gr.Row():
                    num_frames = gr.Number(value=231, label="帧数", precision=0, minimum=33, maximum=231)
                    num_inference_steps = gr.Slider(1, 10, value=2, step=1, label="每阶段步数")
                with gr.Row():
                    seed = gr.Number(value=-1, label="随机种子（-1为随机）", precision=0)
                    is_amplify_first_chunk = gr.Checkbox(label="增强首段", value=True)

            generate_btn = gr.Button("🚀 开始生成", variant="primary", size="lg")

        with gr.Column(scale=1):
            video_output = gr.Video(label="生成的视频", autoplay=True)
            info_output = gr.Textbox(label="生成信息", interactive=False)

    mode.change(fn=update_conditional_visibility, inputs=[mode], outputs=[image_input, video_input])
    resolution.change(fn=update_resolution, inputs=[resolution], outputs=[height, width])
    generate_btn.click(
        fn=generate_video,
        inputs=[
            mode,
            prompt,
            image_input,
            video_input,
            height,
            width,
            num_frames,
            num_inference_steps,
            seed,
            is_amplify_first_chunk,
        ],
        outputs=[video_output, info_output],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=CSS,
        theme=gr.themes.Soft()
    )
