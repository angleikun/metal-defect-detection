# Metal Surface Defect Detection

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-orange.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/YOLOv8-8.4.60-green.svg)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

基于 NEU-DET 数据集和 YOLOv8 框架的工业金属表面缺陷检测系统。

## 项目目标

对钢板、汽车冲压件、铝型材等金属表面的 6 类缺陷（crazing / inclusion / patches / pitted_surface / rolled-in_scale / scratches）进行自动检测和分类，提供桌面应用和 ONNX 推理部署。

## 技术栈

| 层次 | 技术 |
|------|------|
| 深度学习框架 | PyTorch 2.5 + CUDA 12.1 |
| 检测模型 | YOLOv8（Ultralytics 8.4.60） |
| 分割对照 | U-Net（segmentation_models_pytorch） |
| 图像处理 | OpenCV 4.13 |
| 桌面应用 | PyQt6 6.6.1 + qfluentwidgets |
| 推理部署 | ONNX Runtime |
| 实验追踪 | TensorBoard + CSV |
| 硬件 | RTX 3060 Laptop 6GB |

## 项目结构

```
metal-defect-detection/
├── data/
│   ├── raw/NEU-DET/          # 数据集（不提交 git）
│   ├── splits/                # train/val/test 划分文件
│   └── annotations_manual/    # 人工标注（30 张 test mask）
├── notebooks/                 # Jupyter 探索笔记
│   ├── 01_eda.ipynb           # 数据探索
│   └── 02_eval_and_bad_case.ipynb  # 评测与坏样本分析
├── src/
│   ├── utils/
│   │   ├── seed.py            # 随机种子统一设置
│   │   └── __init__.py
│   ├── split_dataset.py       # 数据集分层划分
│   ├── convert_voc_to_yolo.py # VOC → YOLO 格式转换
│   └── ...
├── models/                    # 模型权重（不提交 git）
├── configs/
│   └── data.yaml              # YOLOv8 数据配置
├── docs/                      # 技术文档
├── CONSTRAINTS.md             # 技术约束（Claude Code 必读）
├── PROGRESS.md                # 每日进度跟踪
├── PROJECT_BRIEF.md           # 项目摘要
├── PROJECT_PLAN.md            # 完整执行计划
├── DEVLOG.md                  # 开发日志
├── requirements.txt           # Python 依赖
└── README.md                  # 本文件
```

## 快速开始

### 1. 环境准备

**重要：所有项目命令必须先激活 pytorch 环境。**

```powershell
# 方式 A：手动激活
mamba activate pytorch

# 方式 B：运行项目脚本
activate_env.bat
```

```powershell
# 验证环境
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "from PyQt6.QtWidgets import QApplication; print('PyQt6 OK')"
```

# 安装项目依赖
pip install -r requirements.txt

# 验证 GPU 可用
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 2. 下载数据集

NEU-DET 数据集（约 100MB），二选一：

- **Kaggle（推荐）**：https://www.kaggle.com/datasets/zhangyunsheng/defects-class-and-location
- **东北大学官方**：http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm

下载后解压到 `data/raw/NEU-DET/`，目录结构应为：

```
data/raw/NEU-DET/
├── IMAGES/          # 1800 张图像
└── ANNOTATIONS/     # 1800 个 XML 标注
```

### 3. 训练

```powershell
# 第一次训练前，遵守 CONSTRAINTS.md 的约束
python train.py
```

## TODO

- [x] 项目初始化 + 目录结构
- [ ] Day 1: 下载 NEU-DET 数据集
- [ ] Day 2: 数据探索（EDA）
- [ ] Day 2.5: 固定数据集 split
- [ ] Day 3: YOLOv8 训练跑通
- [ ] Day 4: 调参 + 数据增强
- [ ] Day 5: 评测 + Bad Case 分析
- [ ] Day 5.5: 人工标注 30 张 test mask
- [ ] Week 2: U-Net 分割 + 多模型对比
- [ ] Week 3: PyQt6 桌面应用
- [ ] Week 4: ONNX 部署 + GitHub 上线

## 验收指标

| 指标 | 最低 | 目标 | 拉满 |
|------|------|------|------|
| mAP@0.5（整体） | 0.80 | 0.85 | 0.88 |
| mAP@0.5:0.95 | 0.40 | 0.50 | 0.55 |
| 每类 Recall | 0.75 | 0.85 | 0.90 |
| 推理速度（PyTorch GPU）| < 50ms | < 25ms | - |
| 推理速度（ONNX GPU）| < 30ms | < 15ms | - |

## 许可

MIT License

## 作者

linsanqin — [github.com/angleikun](https://github.com/angleikun)
