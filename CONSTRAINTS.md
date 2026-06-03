# 本项目技术约束（Claude Code 必读）

## 通用原则
1. 不要一上来生成完整框架代码。先写一份"我打算怎么做"的清单，
   人确认后再分步实现。
2. 单次代码量 ≤ 200 行。超过的拆成多个步骤。
3. 每写完一段代码，自己 review 一遍，列出"这段代码可能在哪里出错"。
4. 涉及深度学习训练时，第一次必须用 ≤5 个 epoch 跑通流程，
   而不是直接训练 100 epoch。

## Python / PyTorch 已知坑
- 随机种子：必须调用 src/utils/seed.py 的 set_seed() 统一封装，不要散写
- numpy 2.x 和某些老库不兼容（pandas <2.0 / opencv-python <4.8 会报错）
- CUDA OOM：batch_size 不要超过 16，imgsz 不要超过 640（RTX 3060 6GB）
- DataLoader num_workers 见"num_workers 实测原则"章节，不写死成 0

## YOLOv8（Ultralytics）已知坑
- NEU-DET 图像尺寸 200×200，必须显式设 imgsz=640（默认 640 但要确认）
- 训练前必须验证 data.yaml 的 path/train/val 路径是绝对路径
- model.train() 默认会启动 wandb，第一次跑会卡，要么登录要么 wandb mode=disabled
- save_period 默认 -1 不存中间 checkpoint，长训练前要改
- 训练结果在 runs/detect/train*/ 目录，多次训练会自动加序号

## OpenCV 已知坑
- cv2.imread() 不支持中文路径！用 cv2.imdecode + np.fromfile 替代
- BGR vs RGB：OpenCV 是 BGR，PIL/PyTorch/matplotlib 是 RGB
- imshow 在 Jupyter 里会闪退，用 matplotlib 显示

## PyQt6 已知坑（继承自上一个项目）
- 跨线程访问 GUI 必须 invokeMethod + QueuedConnection
- QMutex 不可递归
- 信号槽默认 AutoConnection，跨线程时要显式指定
- 见 D:\dev_notes\AI_CODING_TRAPS.md

## 实验可复现性（强制）

每次训练前必须调用统一的 set_seed 工具。位置：src/utils/seed.py。

```python
def set_seed(seed: int = 42) -> None:
    """三件套封装。所有训练、评测、数据划分脚本统一调用。

    注意：故意不开 torch.use_deterministic_algorithms(True)。
    ultralytics 8.x 内部 scatter_add 等 op 不支持，开了直接报错训不动。
    cudnn.deterministic + manual_seed 已足够，实测同 seed 重跑
    mAP 浮动 < 0.005。

    Claude Code 后续如果"优化"代码想加 use_deterministic_algorithms，
    必须 review 这条注释，不要加。
    """
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

YOLOv8 训练调用必须显式传：
- `seed=42`
- `deterministic=True`

注意 ultralytics 8.4 在 deterministic=True 下 mosaic/mixup 的随机性
仍受 numpy seed 控制，所以 np.random.seed 不能漏。

DEVLOG 至少记一次"同 seed 重跑 mAP 浮动 < 0.005"作为可复现性实证。

## num_workers（实测原则，不写死）

- Day 3 首次训练用 num_workers=0 跑通
- 跑通后再测 num_workers=2、4，记录单 epoch 耗时
- 选稳定且最快的一档，写进 DEVLOG
- 不要默认用 0；性能损失实测在 25-40%

## PyQt6 版本锁定（Week 3 安装时）

```
pip install PyQt6==6.6.1 PyQt6-Fluent-Widgets==1.5.7
```

PyQt6 6.7+ 和 qfluentwidgets 偶有兼容问题，2025 Q4 这两个版本组合最稳。

## LabelMe 使用规范（Day 5.5 人工标注用）

- 工具：必须 polygon，不要用矩形/圆形（IoU 评测要求像素级精度）
- 类名严格用 NEU-DET 原名（小写 + 下划线）：
  `crazing` / `inclusion` / `patches` / `pitted_surface` / `rolled-in_scale` / `scratches`
- 路径全程不能有中文
- 启动命令见 PROJECT_PLAN §四 Day 5.5（唯一权威源，本节不重复）
- --nodata 选项：JSON 不嵌入图像数据（文件小，git diff 友好）
- 标完后必须跑 src/labelme_to_mask.py 生成 PNG mask
- JSON 和 PNG 都进 git，路径见 .gitignore 放行规则

## 数据集类名命名（重要，避免笔误）

NEU-DET 数据集真实类名：
`crazing` / `inclusion` / `patches` / `pitted_surface` / `rolled-in_scale` / `scratches`

注意 crazing（裂纹结构），不是 "Crack"。原 v1.0 文档里的 "Crack" 是笔误。
所有代码、数据 yaml、文档统一用 NEU-DET 原始小写类名。

## ONNX 已知坑
- PyTorch → ONNX 动态 batch 必须显式 dynamic_axes
- onnxruntime-gpu 和 onnxruntime 不能同时装
- CUDA Provider 要单独装 CUDA 库版本

## 文件路径
- 所有路径用正斜杠 / 或者 raw string r"D:\..."
- 不要写 D:\... 这种字符串（\ 会被转义）
- 项目根目录用 Path(__file__).parent 推导，不要硬编码

## 不要做的事
- 不要把数据集（data/）提交到 git
- 不要把模型权重（*.pt, *.onnx）提交到 git
- 不要在代码里硬编码本地路径
- 不要禁用警告（warnings.filterwarnings 是上一个项目典型 AI 错误）
