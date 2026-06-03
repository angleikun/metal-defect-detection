# 金属表面缺陷检测项目 · 完整执行计划

> **文档用途**：项目从启动到完成的全流程指南。下一次开启新对话时，把整份文档作为上下文发给 Claude，新对话能无缝接手任意阶段的工作。
>
> **作者**：linsanqin（GitHub: angleikun）
> **项目周期**：4 周
> **创建日期**：2026-06-02
> **基于经验**：从上一个工业机器人视觉引导项目（21 个 bug 修复史）提炼的方法论

---

## 一、项目背景与目标

### 1.1 上一个项目的回顾

上一个项目（工业机器人视觉引导系统）使用 Qt + HALCON + OpenCV，做"找物体位置"。21 个真实 bug 的修复经历沉淀在 `D:\dev_notes\AI_CODING_TRAPS.md` 和那个项目的 `DEVLOG.md` 里。

**最大教训**：上来就让 AI 写完整框架 → 必然返工修 N 个 bug。

### 1.2 本项目定位

| 维度 | 上一个项目 | 这个项目 |
|---|---|---|
| 任务 | 定位（找在哪） | 检测（判好坏） |
| 算法 | HALCON 形状匹配 | YOLOv8 + U-Net 深度学习 |
| 数据 | 实时单帧 | 批处理图集 |
| 输出 | 坐标 + 角度 | 类别 + 框 + 评分 |
| 部署 | Qt 桌面应用 | PyQt6 桌面 + ONNX 推理 |

**两个项目互补**，简历呈现"工业视觉全栈：底层 HALCON + 上层深度学习"。

### 1.3 简历可写的最终产出物

```
✓ NEU-DET 数据集上 mAP@0.5 = XX%（实测）
✓ 单图推理 GPU XXms / CPU XXms（实测）
✓ 6 类缺陷分类 + 像素级分割双任务
✓ PyQt6 桌面应用，批量处理 + 报表导出
✓ ONNX Runtime 部署，无需 PyTorch 依赖
✓ 完整 DEVLOG 记录工程开发过程
```

---

## 二、技术栈（已确定）

```
Python           3.12（miniforge 自带，已装）
PyTorch          2.5.1 + CUDA 12.1（已装，复用 pytorch 环境）
Ultralytics      8.4.60（YOLOv8 主框架）
OpenCV           4.13（图像处理）
NumPy/Pandas     2.x / 2.x（数据处理）
matplotlib       3.10（可视化）
PyQt6            待装（Week 3）
ONNX Runtime     待装（Week 4）
torchmetrics     待装（评测指标）

GPU              RTX 3060 Laptop 6GB（够用）
平台             Windows 11
```

### 数据集

- **NEU-DET**（东北大学钢板表面缺陷数据集）
- 6 类：crazing / inclusion / patches / pitted_surface / rolled-in_scale / scratches
- 1800 张图（每类 300 张），含 PASCAL VOC 格式标注
- 下载地址（候选）：
  - 官方：http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm
  - Kaggle：https://www.kaggle.com/datasets/zhangyunsheng/defects-class-and-location（推荐）
  - GitHub 镜像：搜索 "NEU-DET" 找最新可用源

---

## 三、必须先做的事（开工前 1 小时）

### 3.1 写 `CONSTRAINTS.md`（30 分钟）

**作用**：把已知陷阱写成文档喂给 Claude Code，避免它重蹈覆辙。下面是项目根目录的 `CONSTRAINTS.md` 初稿：

```markdown
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
```

### 3.2 准备项目目录

```powershell
# 进入项目存放位置
cd /d D:\Mycode\projects

# 建项目目录
mkdir metal-defect-detection
cd metal-defect-detection

# 初始化 git
git init
git config user.name "linsanqin"
git config user.email "149247650+angleikun@users.noreply.github.com"

# 建基础目录
New-Item -Path "data","notebooks","src","models","docs","configs" -ItemType Directory -Force
New-Item -Path "data\raw","data\splits","src\utils" -ItemType Directory -Force
New-Item -Path "data\annotations_manual\images","data\annotations_manual\jsons","data\annotations_manual\masks" -ItemType Directory -Force

# 建 .gitignore（避免提交数据和模型）
@'
# Data and models (NEVER commit raw data)
data/raw/
models/
*.pt
*.onnx
*.engine

# 放行 splits（实验可复现性必需）
!data/splits/
!data/splits/*.txt
# 放行人工标注（U-Net 评测基准，必须 commit）
!data/annotations_manual/
!data/annotations_manual/**

# Python
__pycache__/
*.pyc
.ipynb_checkpoints/
*.egg-info/

# Training outputs
runs/
wandb/
*.log

# IDE
.vscode/
.idea/

# OS
Thumbs.db
.DS_Store

# Local secrets
.env
local_*.json
'@ | Out-File -FilePath .gitignore -Encoding UTF8

# 验证
type .gitignore
dir
```

### 3.3 选好面试目标方向

不同方向影响 Week 2-4 的侧重点：

- [ ] **海康/大华**（传统视觉 + 工业部署）→ 重 ONNX 性能、重 C++ 推理
- [ ] **思谋/阿丘**（AI 视觉新势力）→ 重模型创新、重 demo 视频
- [ ] **大厂工业方向**（华为/字节）→ 重整体工程能力
- [ ] **学校实验室**（科研方向）→ 重消融实验、重论文阅读笔记
- [ ] **通用方向**（保持灵活性）→ 平衡发力

**先选一个**，写在项目 README 的备注里。

---

## 四、Week 1：算法跑通（Day 1-5）

> **本周目标**：终端一行命令完成检测，输出带框图像。mAP@0.5 > 0.85。

### Day 1：项目初始化 + 数据下载（2-3 小时）

#### 任务清单

- [ ] 项目目录建好（见 §3.2，包含 data/splits/ 和 data/annotations_manual/{images,jsons,masks} 和 src/utils/）
- [ ] CONSTRAINTS.md 写好（见 §3.1，含可复现性 / num_workers / PyQt6 / LabelMe / 类名规范 五节）
- [ ] requirements.txt 锁定所有主要库版本：
      ```
      ultralytics==8.4.60
      numpy==2.2.6
      opencv-python==4.13
      pandas==2.3.3
      matplotlib==3.10.9
      labelme==5.5.0
      pycocotools==2.0.7
      segmentation-models-pytorch==0.3.4
      ```
- [ ] 写 src/utils/seed.py（参考 CONSTRAINTS.md 中的 set_seed 模板）
- [ ] 写 src/utils/__init__.py（空文件，让 utils 成为 Python 包）
- [ ] 写 README.md 第一稿（项目目标 + 技术栈 + 复现说明）
- [ ] 下载 NEU-DET 数据集
- [ ] 解压到 `data/raw/NEU-DET/` 目录
- [ ] 写 Day 1 日志到 DEVLOG.md

#### Claude Code 启动 Prompt（Day 1 用）

```
项目背景：金属表面缺陷检测。技术栈 PyTorch 2.5 + YOLOv8 + RTX 3060。
项目目录：D:\Mycode\projects\metal-defect-detection
请先读 CONSTRAINTS.md 后再回答。

Day 1 任务：
1. 检查项目目录结构是否完整（data/ notebooks/ src/ models/ docs/ configs/）
2. 检查 .gitignore 是否包含 data/, models/, *.pt, runs/
3. 如果 data/raw/NEU-DET/ 不存在，告诉我如何下载（不要自己下，告诉我步骤）
4. 帮我写 README.md 第一稿，包含：项目目标、技术栈、目录结构、TODO 列表
5. 写 src/utils/seed.py（参考 CONSTRAINTS.md 中的 set_seed 模板）
   照搬即可，不算"算法代码"。
6. 写 src/utils/__init__.py（空文件，让 utils 成为 Python 包）
7. 写 requirements.txt 锁定所有主要库版本（见任务清单）

不要执行训练。不要写训练/评测/推理代码。仅做组织 + 工具函数。
```

#### NEU-DET 数据集获取（建议方案）

**方案 A**：从 Kaggle 下载
```
1. 浏览器打开 kaggle.com/datasets/zhangyunsheng/defects-class-and-location
2. 注册 / 登录
3. 点击 Download 按钮（约 100MB）
4. 解压到 data/raw/NEU-DET/
```

**方案 B**：用 Python 自动下载（需要 Kaggle API token）
```python
# 等 Claude Code 走完 Day 1 流程后再用这条
import kagglehub
path = kagglehub.dataset_download("zhangyunsheng/defects-class-and-location")
```

#### Day 1 验收标准

```
data/raw/NEU-DET/
├── IMAGES/                  # 1800 张 .jpg 或 .bmp
└── ANNOTATIONS/             # 1800 个 .xml（PASCAL VOC 格式）
```

或者其他类似结构（具体看下载源）。

---

### Day 2：数据探索（EDA，3-4 小时）

#### 任务清单

- [ ] 用 Jupyter Notebook 探索数据
- [ ] 统计每类缺陷的图像数量
- [ ] 可视化每类的 3-5 个样本
- [ ] 输出 bbox 面积分布直方图（原图坐标系，200×200）
- [ ] 用 COCO 阈值 32²/96² 把 bbox 分成 small/medium/large 三档
- [ ] 统计三档样本数
- [ ] 若某档 bbox 数 < 30：DEVLOG 记一行"该档 mAP 高方差，简历仅参考不强报"
      （不需要改 BRIEF/PLAN 文档本身，只在 DEVLOG 留档）
- [ ] 检查图像尺寸是否一致
- [ ] 写 `notebooks/01_eda.ipynb`

#### Claude Code Prompt（Day 2）

```
请帮我写 notebooks/01_eda.ipynb 进行数据探索。

要求：
1. 使用相对路径 ../data/raw/NEU-DET（不要硬编码 D:\）
2. 分 6 个 cell 渐进式完成：
   Cell 1: import 所需库 + 设置路径
   Cell 2: 加载所有标注 XML，统计每类缺陷数量
   Cell 3: 画出每类数量的柱状图
   Cell 4: 随机抽 6 张图（每类 1 张）显示原图 + 标注框
   Cell 5: 统计所有标注框的尺寸分布（宽、高、面积、宽高比）
   Cell 6: 输出结论摘要

3. 注意 OpenCV 中文路径问题，用 cv2.imdecode 读图

完成后告诉我每个 cell 的预期输出形状，让我自己跑。
不要自己跑（沙盒没有数据）。
```

#### Day 2 验收标准

Notebook 里能看到：
- 6 类缺陷数量柱状图（应该都是 300 张左右）
- 每类的样本可视化（图 + 标注框）
- 标注框尺寸的直方图
- 三档（small/medium/large）的 bbox 样本数统计
- 一段总结文字写在最后一个 Markdown cell

---

### Day 2.5：固定数据集 split（1 小时，关键步骤）

EDA 完成后立刻做 split，避免 Day 3 训练前临时拍。

#### 任务清单

- [ ] 写 src/split_dataset.py
- [ ] 按类分层（stratified by class），seed=42
- [ ] 80 / 10 / 10 划分（1440 / 180 / 180）
- [ ] 落成静态文件：data/splits/{train,val,test}.txt
- [ ] 每行一个图像 stem（不含扩展名）
- [ ] git add data/splits/ 后立刻 commit（注意 .gitignore 不要漏放行）

#### Claude Code Prompt

```
请写 src/split_dataset.py，做 NEU-DET 数据集的分层 split。

约束：
1. 用 src/utils/seed.py 的 set_seed(42)
2. 按类分层：6 类各自 80/10/10 划分，避免类不平衡
3. 输出 data/splits/train.txt、val.txt、test.txt
4. 每行一个图像 stem（不含 .jpg 后缀）
5. 脚本幂等：重复运行结果必须完全一致
6. 脚本结尾打印每个 split 的类分布（验证分层正确）
7. 所有文件列表必须 sorted() 后再 split。
   os.listdir() 跨机器/Python 版本顺序不同，即使 seed=42 也会出
   不一致的 split。这条不可省。
8. 写完后给 train.txt 跑 sha256，记到 DEVLOG。
   后续任何对 split 的"误改动"都能通过对比 sha256 发现。

完成后让我跑，确认三个 txt 行数为 1440/180/180，且每类比例都是 4:0.5:0.5。
```

#### Day 2.5 验收

- data/splits/train.txt（1440 行）
- data/splits/val.txt（180 行）
- data/splits/test.txt（180 行）
- 三个文件已 git commit
- DEVLOG 记一行："Day 2.5 完成 stratified split，每类 240/30/30，train.txt sha256 = xxx"

---

### Day 3：YOLOv8 训练跑通（4-5 小时）

#### 任务清单

- [ ] 把 NEU-DET 转成 YOLO 格式（XML → TXT）
- [ ] 写 `configs/data.yaml`
- [ ] 第一次训练（5 epoch 验证流程）
- [ ] 看训练日志，确认没报错
- [ ] 跑完整 50 epoch
- [ ] 在验证集上测 mAP

#### Claude Code Prompt（Day 3）

```
请帮我做 NEU-DET 数据集的 YOLOv8 训练。

约束：
1. 严格遵守 CONSTRAINTS.md
2. 数据集格式转换：NEU-DET 是 PASCAL VOC，YOLOv8 要 YOLO format
3. 分阶段：
   Step 1: 写转换脚本 src/convert_voc_to_yolo.py
   Step 2: 写 configs/data.yaml
   Step 3: 写 train.py（用 ultralytics.YOLO）
   Step 4: 先跑 5 epoch 验证流程
   Step 5: 跑完整 50 epoch

每一步完成后停下来给我看结果。

特别注意：
- imgsz 显式设 640（NEU-DET 原图 200x200 会被放大）
- batch_size 16（6GB 显存够）
- device 必须显式设 0（GPU），不要让它自动选
- workers 先设 0 跑通，跑通后实测 2 和 4，选稳定且快的（见 CONSTRAINTS）
- 训练前调用 src/utils/seed.py 的 set_seed(42)
- model.train() 必须显式传 seed=42, deterministic=True
- 数据源：configs/data.yaml 必须指向 data/splits/train.txt 和 val.txt
  （不是整个目录！）不允许 ultralytics 默认随机 split
- test.txt 留到 Day 5 评测用，训练期不碰

不要直接训练，先生成全部脚本让我 review。
```

#### Day 3 验收标准

```
models/
└── train_run_1/                    # YOLOv8 自动生成
    ├── weights/
    │   ├── best.pt                 # 最佳权重
    │   └── last.pt                 # 最后一轮
    └── results.csv                 # 训练曲线数据
```

mAP@0.5 应该 ≥ 0.7（基线 YOLOv8 不调参的默认水平）。

---

### Day 4：调参 + 数据增强（4-5 小时）

#### 任务清单

#### Day 4 数据集口径（严格遵守）

- [ ] 本周（Day 4）所有 mAP 数字均在 val set（180 张）上
- [ ] Day 5 才在 test set（180 张）上跑一次终值
- [ ] test set 在 Day 4 期间完全不碰，包括"瞄一眼"
- [ ] DEVLOG 里所有 mAP 数字必须标注口径（val/test）
- [ ] README/简历最终数字必须是 test set 数字

#### Day 4 复现性自检（半小时，一次性）

- [ ] 选定 baseline 配置后，用同 seed=42 完整重跑一次
- [ ] 对比两次 val mAP@0.5，浮动应 < 0.005
- [ ] 若 > 0.005：检查 set_seed 是否漏了某项、deterministic 是否开
- [ ] DEVLOG 记两次 mAP 数字 + 浮动值（这是 README 验收表"复现性"行的来源）

#### Day 4 任务清单（原内容）

- [ ] 调整数据增强参数（mosaic / mixup / hsv）
- [ ] 试不同 backbone（n / s / m）对比速度和精度
- [ ] 加 cosine LR scheduler
- [ ] 跑 100 epoch
- [ ] 目标 val mAP@0.5 > 0.85

#### Claude Code Prompt（Day 4）

```
基于 Day 3 的训练结果，做参数优化。

约束：
- 一次只改一个参数（控制变量）
- 每次实验都要在 DEVLOG.md 记一行（实验 ID + 改了什么 + 结果）
- 总共做 3-5 次实验，不要超过

实验列表：
1. baseline (Day 3 的结果)
2. yolov8n → yolov8s (更大模型)
3. 关闭 mosaic 数据增强
4. 加 cosine LR
5. 增加训练轮数到 100

每个实验跑完，把 results.csv 的最终 mAP 写进 DEVLOG。
最后选 mAP 最高的那个作为 best 模型。
```

#### Day 4 验收标准

`DEVLOG.md` 包含一个实验对比表：

| 实验 | 模型 | val mAP@0.5 | val mAP@0.5:0.95 | 训练时间 |
|---|---|---|---|---|
| baseline | yolov8n | 0.78 | 0.45 | 15min |
| ... | ... | ... | ... | ... |

---

### Day 5：评测 + Bad Case 分析（3-4 小时）

#### 任务清单

- [ ] 在 test set（不是 val！）上跑混淆矩阵
- [ ] 找出每类的 P/R/F1
- [ ] 按 COCO 口径分尺度评测 mAP@0.5（small <32² / medium 32²-96² / large >96²）
- [ ] 画分尺度 mAP 柱状图（6 类 × 3 尺度）
- [ ] 挑 10 个最难样本（IoU 低 / 分类错）
- [ ] 分析失败原因（光照？尺度？标注误？）—— 特别关注 small 目标的失败模式
- [ ] 写 `notebooks/02_eval_and_bad_case.ipynb`
- [ ] DEVLOG 记一段"分尺度 insight"：哪一类在哪个尺度上最弱，归因

#### Claude Code Prompt（Day 5）

```
请帮我写 notebooks/02_eval_and_bad_case.ipynb。

约束：
- 用 best.pt 模型（来自 Day 4）
- 在完整验证集上预测，保存所有结果
- 计算每类 P/R/F1，画混淆矩阵
- 找出 IoU < 0.5 的样本，按错误程度排序
- 选 10 个最难样本可视化（原图 + GT 框 + 预测框）
- 每个 bad case 给一句失败原因猜测

完成后让我跑，跑完结果记到 DEVLOG。
```

### Day 5.5：人工标注 30 张 test mask（3-4 小时，跨 Day 5 晚 → Day 6 早）

> 这是 U-Net 评测能"扛追问"的关键。看一集电视剧的时间就能完成。
> 注意：这是 U-Net 评测的前置任务，必须在 Day 6 训练前完成。

#### 任务清单（按顺序执行）

- [ ] 写 src/select_annotation_samples.py
- [ ] 跑脚本：抽 30 张 + copy 原图到 data/annotations_manual/images/
- [ ] 启动 LabelMe（从 images/ 加载，JSON 写到 jsons/）
- [ ] 逐张标 polygon（30 张）
- [ ] 写 src/labelme_to_mask.py
- [ ] 跑脚本：jsons/ → masks/（30 个 PNG）
- [ ] 验证：肉眼抽看 3 张 mask 是否合理
- [ ] git add data/annotations_manual/ && commit
- [ ] git check-ignore -v 验证三个子目录都未被忽略

#### 目录结构（标完后）

```
data/annotations_manual/
├── samples.txt              # 30 个图像 stem（不含扩展名）
├── images/                  # 30 张原图副本（标注源，进 git）
│   ├── crazing_001.jpg
│   └── ...
├── jsons/                   # LabelMe 30 个 JSON 标注
│   ├── crazing_001.json
│   └── ...
└── masks/                   # 30 个 PNG mask（0=bg, 255=defect）
    ├── crazing_001.png
    └── ...
```

注意 images/ 是必需的——LabelMe 必须从图片目录启动，
NEU-DET 原始目录 1800 张不能让它全加载。
先 copy 30 张到独立目录，再启 LabelMe。

#### Claude Code Prompt（写两个脚本）

```
请写两个脚本：

1. src/select_annotation_samples.py
   - 用 src/utils/seed.py 的 set_seed(42)
   - 读 data/splits/test.txt（180 张）
   - 每类随机抽 5 张（np.random.choice，不要按分位数挑——过度设计）
   - 写 data/annotations_manual/samples.txt（30 行 stem）
   - 把这 30 张原图从 data/raw/NEU-DET/IMAGES/ 复制到
     data/annotations_manual/images/（shutil.copy2 保留时间戳）
   - 幂等：若目标文件已存在则 skip
   - 结尾打印每类样本数（都应该是 5）

2. src/labelme_to_mask.py
   - 输入：data/annotations_manual/jsons/*.json
   - 输出：data/annotations_manual/masks/*.png
   - mask 单通道 uint8，0=背景 255=缺陷
   - polygon 用 cv2.fillPoly
   - 多边形重叠取并
   - mask 尺寸 = 对应原图尺寸（读 JSON 的 imageHeight/imageWidth 字段）
   - 完成后打印每张 mask 的前景像素占比，确认都 ≥ 0.5%
```

#### LabelMe 操作流程（一次性走通）

```powershell
# 1. 跑选样脚本（10 秒，生成 samples.txt + copy 30 张图）
python src\select_annotation_samples.py

# 2. 启动 LabelMe GUI
labelme data\annotations_manual\images --output data\annotations_manual\jsons --autosave --nodata

# 3. 在 GUI 里：每张图画 polygon + 选类名（crazing/inclusion/...）
#    autosave 模式下 JSON 自动保存到 --output 目录

# 4. 全部标完关 GUI，跑转换脚本
python src\labelme_to_mask.py
```

#### 标注规范

- 工具：polygon 多边形（不要矩形/圆形/圆圈）
- 类名：严格按 NEU-DET 6 类小写名：
  `crazing` / `inclusion` / `patches` / `pitted_surface` / `rolled-in_scale` / `scratches`
- 一张图可能多个缺陷，每个画一个 polygon
- 边界尽量贴紧缺陷，宁紧勿松（IoU 评测对边界敏感）

#### Day 5.5 验收标准

- [ ] data/annotations_manual/images/ 有 30 张 jpg
- [ ] data/annotations_manual/jsons/ 有 30 个 JSON
- [ ] data/annotations_manual/masks/ 有 30 个 PNG
- [ ] 每类 5 个文件，文件名按 NEU-DET 类名前缀分布均匀
- [ ] 每个 mask 前景像素占比 ≥ 0.5%
- [ ] git commit 已完成，git check-ignore 验证三个子目录都未被忽略
- [ ] DEVLOG 记一段：标注耗时、遇到的问题（例如某些 crazing 边界模糊难标）

#### 时间预算

- 写两个脚本：30 分钟
- LabelMe 安装 + 熟悉：15 分钟
- 标 30 张（200×200 polygon）：~3-5 分钟/张 × 30 = 90-150 分钟
- 转换 + 验证 + commit：30 分钟
- 总计 3-4 小时

可分两次做：Day 5 晚写脚本 + 标 15 张，Day 6 早标剩 15 张 + 转换。

---

#### Week 1 结束自检

- [ ] 终端一行命令能完成检测：`python detect.py --source xxx.jpg`
- [ ] mAP@0.5 ≥ 0.85
- [ ] DEVLOG.md 至少有 5 条详细记录
- [ ] notebooks/ 下至少 2 个 Notebook
- [ ] **第一次 git commit + push 到 GitHub**

---

## 五、Week 2：U-Net 分割 + 多模型对比（Day 6-10）

> **本周目标**：用 U-Net 做像素级缺陷分割，做工程决策"为什么选 YOLO 不选 U-Net"。

### Day 6-7：U-Net 实现（8-10 小时）

#### 任务清单

- [ ] 前置：Day 5.5 人工标注 30 张已完成（见 §四 Day 5.5）
- [ ] U-Net 模型代码（segmentation_models_pytorch 库）
- [ ] 生成两套 mask：baseline（bbox 内全前景）+ refined（Otsu+形态学）
- [ ] 训练两次 U-Net：一次用 baseline mask，一次用 refined mask
- [ ] 评测均 vs 30 张人工 mask（绝对基准）
- [ ] 对比两次 IoU，量化 refine mask 的价值

#### Claude Code Prompt

```
请实现 NEU-DET 数据集的 U-Net 像素级分割。
本任务的核心是：弱监督伪 mask 生成 + refine + 训练 + 对比。

约束：
- 用 segmentation_models_pytorch 库
- backbone: resnet34
- 复用 data/splits/ 的 train/val/test 划分（不要重新 split）
- 训练 50 epoch
- 评测 GT = data/annotations_manual/masks/ 的 30 张人工 mask

分步骤（重要，按顺序）：

Step 1: 写 src/mask_generator.py
  - 输入：原图 + PASCAL VOC bbox
  - 步骤 a：bbox 内截取 ROI
  - 步骤 b：灰度化 + Otsu 自适应阈值
  - 步骤 c：形态学开运算（去噪）+ 闭运算（连通）
  - 步骤 d：把 ROI 内的二值结果回填到全图 mask
  - 输出：单通道 PNG mask（0=背景，1=缺陷）
  - 同时生成 baseline mask（bbox 内全 1，作对照）

Step 2: 可视化对比 baseline mask vs refined mask
  - 每类抽 3 张图，画原图 + bbox + baseline mask + refined mask 四联图
  - 主观判断 refined mask 是否合理（特别看 crazing 和 scratches）
  - 不合理就回 Step 1 调参数（Otsu 阈值偏置、形态学 kernel size）

Step 3: 写 dataset 类
  - 读 splits/train.txt
  - 读对应的 refined mask 文件
  - 标准化、resize

Step 4: 写训练循环（纯 PyTorch，不用 lightning，避免额外依赖）
  - 用 src/utils/seed.py 的 set_seed
  - loss = 0.5 * BCE + 0.5 * Dice（起点）
  - Dice 用 smooth=1.0，防止 0/0
  - 若训练 20 epoch 后 val IoU < 0.40，调成 0.3 * BCE + 0.7 * Dice
    （NEU-DET 缺陷前景占比 < 10%，纯 BCE 会被背景压制）
  - loss 权重作为超参写进 DEVLOG，不要在代码里硬编码
  - 每 epoch 在 val set 上算 IoU
  - 保存最佳权重到 models/unet_best.pt

Step 5: 评测（vs 30 张人工 mask）
  - 输入：30 张 test 子集（来自 data/annotations_manual/samples.txt）
  - U-Net 预测在 640×640 分辨率
  - 人工 mask 在 200×200 分辨率
  - 评测前必须 resize 预测 mask → 200×200（cv2.INTER_NEAREST，避免插值产生中间值）
  - 二值化阈值 0.5
  - IoU 用 pycocotools 或手写：(pred & gt).sum() / (pred | gt).sum()
  - 同时报：
    * 6 类的 per-class IoU（每类 5 张，方差大，仅参考）
    * 30 张总体 IoU（pixel-level micro 平均，主要指标）
  - 同时跑 baseline mask 训练的 U-Net 在同 30 张上的 IoU 作对照
  - 全部写进 DEVLOG

每步完成停下来 review，特别是 Step 2 必须人眼确认 mask 质量。
```

### Day 8：模型对比实验（4-5 小时）

写一个对比表，三种方法都试：

| 方法 | mAP@0.5 / IoU | 推理速度 | 优点 | 缺点 |
|---|---|---|---|---|
| HALCON 形状匹配 | - | 50ms | 不需数据 | 不能识别新缺陷 |
| YOLOv8 检测 | 0.88 | 15ms | 快、准 | 不给精确轮廓 |
| U-Net 分割 | IoU 0.60（vs 30 张人工 mask） | 80ms | 像素精度 | 慢、依赖标注质量 |

**关键产出**：在 README 写一段"为什么选 YOLO 作为主算法"的工程决策。

### Day 9：消融实验（3-4 小时）

YOLO 的消融实验，证明你做了系统性研究：

**实验约束**（重要）：
- 所有 ablation 必须用同一份 data/splits/train.txt + val.txt
- ablation 之间只允许变一个变量
- 评测在 val set 上比较（不动 test）
- baseline = Day 4 选出的最优配置
- 表格里 "Δ vs baseline" 列必须报相对 baseline 的差值，不是绝对值
- 最终 best 模型在 test set 上跑一次终值，写进 README

| 实验 | 改动 | val mAP@0.5 | Δ vs baseline |
|---|---|---|
| 不用预训练权重 | from scratch | -0.15 |
| 关闭 mosaic | augment.mosaic=0 | -0.04 |
| 关闭 mixup | augment.mixup=0 | -0.01 |
| imgsz 320 | 不放大 | -0.08 |
| imgsz 1280 | 放更大 | +0.02（边际效益低） |

### Day 10：Week 2 总结（2-3 小时）

写 `docs/tech_notes.md`：
- 三种方法对比
- 消融实验结果
- 失败的尝试（这个特别值钱）
- 引用了哪些论文（写 3-5 篇）

---

## 六、Week 3：PyQt6 桌面应用（Day 11-15）

> **本周目标**：能录 demo 视频的完整桌面应用。

### Day 11：界面框架（4-5 小时）

#### 任务清单

- [ ] 装 PyQt6 + qfluentwidgets
- [ ] 写主窗口骨架
- [ ] 左边图像列表，中间预览，右边参数面板

```powershell
pip install PyQt6==6.6.1 PyQt6-Fluent-Widgets==1.5.7

# 说明：PyQt6 6.7+ 和 qfluentwidgets 偶有兼容问题，2025 Q4 这两个版本组合最稳。
# 这两行也已加进项目根 requirements.txt
```

#### Claude Code Prompt

```
请帮我写一个 PyQt6 桌面应用，用于 NEU-DET 缺陷检测的 demo。

约束：
- 使用 qfluentwidgets 库（pip install PyQt6-Fluent-Widgets）
- 复用上一个项目的架构思路：分层（UI / Manager / Algorithm）
- 严格遵守 D:\dev_notes\AI_CODING_TRAPS.md 里的 Qt 多线程规范
- 不要让 inference 在 GUI 线程跑

布局：
- 左侧：图像列表（QListWidget）
- 中间：图像预览 + 标注叠加（自定义 QGraphicsView）
- 右侧：参数面板（置信度阈值、IoU 阈值、模型选择）
- 底部：状态栏 + 处理进度

先只写骨架，不要接模型。完成后我看一下 UI 效果。
```

### Day 12：单图检测 + 可视化（4-5 小时）

#### 任务清单

- [ ] 点击图像 → 显示检测结果
- [ ] 缺陷框 + 类别标签 + 置信度
- [ ] 推理用 worker 线程，GUI 不卡

### Day 13：批处理 + 进度条（3-4 小时）

#### 任务清单

- [ ] 选择文件夹 → 处理所有图
- [ ] 进度条实时更新
- [ ] 处理结果存到列表

### Day 14：报表导出（3-4 小时）

#### 任务清单

- [ ] CSV 导出（用 pandas）
- [ ] Excel 导出（用 openpyxl）
- [ ] PDF 导出（用 reportlab）

**复用上一个项目的 ReportManager 思路。**

### Day 15：设置面板 + 配置持久化（3-4 小时）

#### 任务清单

- [ ] JSON 配置文件
- [ ] 阈值/路径设置可调
- [ ] 启动加载，关闭保存
- [ ] **录 demo 视频**（30 秒-1 分钟）

---

## 七、Week 4：工程化 + 简历包装（Day 16-20）

> **本周目标**：项目能 deploy，简历能直接写。

### Day 16：ONNX 导出（3-4 小时）

```python
# 大致流程
from ultralytics import YOLO
model = YOLO('best.pt')
model.export(format='onnx', dynamic=True, simplify=True)
```

**验证 ONNX 和原模型输出一致**（误差 < 1e-4）。

### Day 17：ONNX Runtime 部署 + 性能基准（4-5 小时）

```powershell
pip install onnxruntime-gpu  # CPU 版用 onnxruntime
```

测试三种推理方式的性能：

| 推理方式 | 单图延迟 | 吞吐量 |
|---|---|---|
| PyTorch GPU | 15ms | 65 img/s |
| ONNX Runtime GPU | 12ms | 80 img/s |
| ONNX Runtime CPU | 180ms | 5.5 img/s |

### Day 18：完整文档（4-5 小时）

#### DEVLOG.md（参照上一个项目格式）

记录至少 10 个真实 bug：
- 数据集路径中文报错
- CUDA OOM
- 训练曲线震荡
- onnx 输出 shape 不对
- ...

#### README.md（按 GitHub 标准写）

模板：

```markdown
# Metal Surface Defect Detection

[badge: license MIT] [badge: Python 3.12] [badge: PyTorch 2.5]

工业金属表面缺陷检测系统，基于 YOLOv8 + PyQt6。

## Demo

![demo](docs/demo.gif)

## 验收测试

| 测试 | 指标 | 结果 | 评测集 |
|---|---|---|---|
| 检测精度（整体）       | mAP@0.5    | 0.XX | test (180) |
| 检测精度（small）      | mAP@0.5    | 0.XX | test (180) |
| 检测精度（medium）     | mAP@0.5    | 0.XX | test (180) |
| 检测精度（large）      | mAP@0.5    | 0.XX | test (180) |
| U-Net 分割             | IoU        | 0.XX | 30 张人工 mask |
| 推理速度（PyTorch GPU）| ms/img     | XX   | test (180) |
| 推理速度（ONNX GPU）   | ms/img     | XX   | test (180) |
| 推理速度（ONNX CPU）   | ms/img     | XX   | test (180) |
| 批处理吞吐             | img/s      | XX   | test (180) |
| 复现性                 | mAP 浮动   | < 0.005 | 同 seed 重跑 2 次 |

## 系统架构

[mermaid 架构图]

## 快速开始

[详细安装步骤]

## 项目结构

[目录树]

## 已知限制

[诚实写出来]
```

### Day 19：GitHub 上传（2-3 小时）

参考上一个项目的 GitHub 上传流程（关键步骤）：

```powershell
cd D:\Mycode\projects\metal-defect-detection

# 确认无敏感信息
git status
type .gitignore  # 确认包含 data/, models/, *.pt 等

# 提交
git add .
git commit -m "Initial commit: metal surface defect detection v1.0"

# GitHub 建空仓库（不勾 README/.gitignore/LICENSE）
# 然后：
git remote add origin https://github.com/angleikun/metal-surface-defect-detection.git
git branch -M main
git push -u origin main

# Topics 加：
# yolov8 deep-learning industrial-inspection pytorch pyqt6 defect-detection neu-det
```

**避免上次的坑**：
- 不要用 PowerShell 改中文内容（用 Python）
- README 编码必须 UTF-8 无 BOM
- 不要 push 数据集和模型权重

### Day 20：简历准备（3-4 小时）

#### 简历项目栏（模板）

```
工业金属表面缺陷检测系统  |  独立开发  |  2026.06
GitHub: github.com/angleikun/metal-surface-defect-detection

技术栈：PyTorch 2.5 / YOLOv8 / U-Net / PyQt6 / ONNX Runtime

- 基于 NEU-DET 数据集训练 YOLOv8 模型，达到 mAP@0.5 = 0.88
- 实现 U-Net 像素级分割作为对照实验，验证 YOLOv8 选型合理性
- 设计基于 PyQt6 + qfluentwidgets 的 Fluent Design 桌面应用，
  支持批量处理 80 img/s
- 使用 ONNX Runtime 部署，相比 PyTorch 推理速度提升 25%
- 完整复盘工程过程，DEVLOG 记录 N 个真实 bug 的诊断和修复
```

#### 准备 3 个 STAR 故事

每个故事按 Situation - Task - Action - Result 结构，能讲 2-3 分钟：

1. **数据集中文路径 bug** 的诊断过程
2. **CUDA OOM 调试** 的过程
3. **PyTorch 和 ONNX 输出不一致** 的排查过程

写在 `docs/interview_stories.md`，背熟。

---

## 八、避坑清单（提炼自上一个项目）

### 沟通层面（最重要）

| 错误做法 | 正确做法 |
|---|---|
| "做个完整的训练系统" | "先写数据加载，让我 review" |
| 让 AI 一次写 500 行 | 单步 ≤ 200 行 |
| AI 报"成功了"就信 | 让 AI 列出"可能哪里出错" |
| 报错堆栈贴一半 | 完整堆栈 + 触发命令 |

### 工程层面

| 项目 | 规则 |
|---|---|
| 路径 | 全用 raw string 或 / |
| 中文路径 | 用 cv2.imdecode + np.fromfile |
| 随机种子 | torch + numpy + random 三个都设 |
| Windows + DataLoader | num_workers=0 |
| 模型保存 | .pt 文件不进 git |
| 数据保存 | data/ 不进 git |

### 文档层面

| 时机 | 该写什么 |
|---|---|
| 每天工作前 | 看 yesterday DEVLOG，明确今天目标 |
| 每天工作后 | 写当天 DEVLOG（耗时 ≤ 15 分钟） |
| 每周末 | 整理 commit history，重写 commit msg |
| 项目末尾 | 把所有 bug 提炼成 STAR 故事 |

---

## 九、Prompt 模板库

### 9.1 启动新功能的标准 Prompt

```
项目：metal-defect-detection
当前阶段：Week X Day Y
之前做完的：[简短列举]
现在要做：[具体目标]

请：
1. 先读项目根目录的 CONSTRAINTS.md
2. 列出你打算做什么（分步骤），不要写代码
3. 我确认后再分步实现，每步不超过 200 行
4. 每步完成后自检：可能在哪里出错？
```

### 9.2 调 bug 的标准 Prompt

```
现象：[完整报错堆栈]
触发命令：[精确命令]
预期：[应该发生什么]
环境：Python 3.12 + PyTorch 2.5 + RTX 3060

请：
1. 列出 3 种可能的根因，按概率排序
2. 对每种给出验证方法
3. 不要直接改代码，先和我讨论
4. 修完写到 DEVLOG.md
```

### 9.3 Code Review 的标准 Prompt（新对话窗口）

```
你是经验丰富的 PyTorch 工程师。下面是我写的代码，请 code review。
特别检查：内存泄漏、CUDA 同步、数据加载效率、复现性、Windows 兼容性。

[贴代码]

输出格式：
| 严重度 | 行号 | 问题 | 建议 |
```

---

## 十、每周总结模板（写到 DEVLOG）

```markdown
## Week X 总结

### 实际产出
- ✅ ...
- ✅ ...
- ⚠️ 计划但没做：...

### 遇到的 bug（按严重度）
- B01: ...
- B02: ...

### 学到的东西
- ...
- ...

### 下周计划调整
- 加：...
- 砍：...

### 简历可写的新点
- ...
```

---

## 十一、紧急情况处理

| 情况 | 处理方式 |
|---|---|
| 训练卡死 | 杀进程，看显存（nvidia-smi），重启 |
| CUDA OOM | 减 batch_size 一半，看是否解决 |
| 数据加载报错 | 先 `dataset[0]` 测试单样本 |
| 模型 mAP 一直不动 | 检查学习率太小、数据增强太强、标签错位 |
| Git push 失败 | 先看 git status，确认没大文件 |
| 进度严重落后 | 砍 Week 2 的 U-Net 部分，Week 3-4 必做 |

---

## 十二、最低验收线

如果时间紧张，**至少完成这些才算项目成功**：

- [ ] YOLOv8 训完，mAP > 0.8
- [ ] GitHub 上有完整代码 + README + LICENSE
- [ ] 录了一段 demo 视频（哪怕只展示终端运行）
- [ ] DEVLOG 至少 5 条真实 bug 记录

砍掉的优先级：U-Net → ONNX → PyQt6 → 报表导出

---

## 十三、文档变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-06-02 | v1.0 | 初版完成 |
| 2026-06-02 | v1.1 | 实验可信度三件套（split/复现性/test 终值）+ U-Net refine mask + 分尺度 mAP + 工程细节微调 |
| 2026-06-02 | v1.1.1 | audit 第二轮：U-Net 评测改 vs 30 张人工 mask 绝对基准 + 数字/口径内部一致性同步 + LabelMe 规范 + 类名笔误修正（Crack → crazing） |

---

## 附录 A：环境检查清单

每次开机后第一件事，确认环境状态：

```powershell
# 1. 进环境
mamba activate pytorch    # 或 defect

# 2. 进项目
cd /d D:\Mycode\projects\metal-defect-detection

# 3. 验证 GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 4. 看 git 状态
git status

# 5. 看 DEVLOG 最后一条
type DEVLOG.md
```

## 附录 B：当前已确认环境信息

```
Python      3.12 (miniforge)
PyTorch     2.5.1+cu121
torchvision 0.20.1+cu121
ultralytics 8.4.60
opencv      4.13
NumPy       2.2.6
pandas      2.3.3
matplotlib  3.10.9
GPU         RTX 3060 Laptop 6GB, Driver 546.92, CUDA 12.3
环境名      pytorch (位于 C:\Users\Admin\miniforge3\envs\pytorch)
```

## 附录 C：相关参考资源

- [Ultralytics 官方文档](https://docs.ultralytics.com)
- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [NEU-DET 论文](https://ieeexplore.ieee.org/document/8019350)（Song et al. 2013）
- [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- 你的私人知识库：`D:\dev_notes\AI_CODING_TRAPS.md`

---

## 附录 D：下一个对话开场白模板

新对话窗口直接发：

```
你好。我在做金属表面缺陷检测项目（4 周计划），技术栈 PyTorch 2.5 + YOLOv8。

完整执行计划在附件 PROJECT_PLAN.md。请先读一遍。

当前状态：Week X Day Y
昨天做完：[列举]
今天要做：[列举]

请先读 PROJECT_PLAN.md 的 §X 章节，然后告诉我今天该怎么做。
注意遵守 §VIII 避坑清单。
```

---

**文档结束**。

预祝项目顺利。下次对话见。
