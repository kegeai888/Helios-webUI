# Helios 项目快速部署

## 一键部署流程

### 步骤 1: 安装依赖
```bash
bash install.sh
```

### 步骤 2: 下载模型
```bash
bash download_models.sh
```
选择 `3) Helios-Distilled` 用于 WebUI（推荐）

### 步骤 3: 启动 WebUI
```bash
bash start_app.sh
```

访问：http://localhost:7860

---

## 详细说明

### 环境要求
- conda (anaconda/miniconda)
- CUDA 兼容的 GPU
- Python 3.8+

### 目录说明
- `models/` - 模型存放（需下载）
- `outputs/` - 生成视频输出
- `app.py` - WebUI 主程序
- `start_app.sh` - 启动脚本

### 模型选择
- **Helios-Distilled**: 速度快，显存占用小（推荐 WebUI）
- **Helios-Mid**: 平衡性能
- **Helios-Base**: 最高质量

### 版权信息
webUI二次开发 by 科哥 | 微信：312088415 公众号：科哥玩AI  
承诺永远开源使用 但是需要保留本人版权信息！

详细文档见：`部署指南.md`
