# DEVLOG

> 每条记录必须包含四段：现象 / 根因 / 修复 / 教训。
> 每天工作结束追加一条当日总结。

---

## Day 1 — 2026-06-02：项目初始化

### 完成事项

- [x] 项目根目录创建（`data/`, `notebooks/`, `src/`, `models/`, `docs/`, `configs/`）
- [x] 子目录创建（`data/raw/`, `data/splits/`, `src/utils/`）
- [x] 人工标注目录预留（`data/annotations_manual/images/`, `jsons/`, `masks/`）
- [x] Git 初始化 + `.gitignore`（排除 `data/raw/`、`models/`、`*.pt`、`*.onnx`、`runs/`）
- [x] `CONSTRAINTS.md` 技术约束文档
- [x] `requirements.txt` 依赖锁定
- [x] `src/utils/seed.py` 可复现性种子封装
- [x] `src/utils/__init__.py` 和 `src/__init__.py`
- [x] `README.md` 第一稿（项目目标 + 技术栈 + 目录结构 + TODO）
- [x] `DEVLOG.md` 本文件

### 待完成（需用户手动操作）

- [ ] 下载 NEU-DET 数据集（~100MB），解压到 `data/raw/NEU-DET/`
  - Kaggle：https://www.kaggle.com/datasets/zhangyunsheng/defects-class-and-location
  - 或东北大学官方：http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm
- [ ] 验证解压后目录结构：`IMAGES/`（1800 张） + `ANNOTATIONS/`（1800 个 XML）

### 环境检查

| 项目 | 状态 |
|------|------|
| Python 3.12 | 待验证（`mamba activate pytorch`） |
| PyTorch 2.5.1 + CUDA 12.1 | 待验证 |
| Ultralytics 8.4.60 | 待安装（`pip install -r requirements.txt`） |
| RTX 3060 6GB | 待验证（`nvidia-smi`） |

### 关键决策

- 面试方向选择：**通用方向**（平衡发力，不加偏重某个行业）
- NEU-DET 下载方案：首选 Kaggle，备选官方页面

### 明日（Day 2）预告

1. 确认数据集已下载解压
2. 开始 EDA：写 `notebooks/01_eda.ipynb`
3. 统计每类样本数、bbox 面积分布、small/medium/large 三档样本数

---

## Day 2 — 2026-06-03：数据探索（EDA）

### 完成事项

- [x] `notebooks/01_eda.ipynb` 已写好（16 cells: 8 code + 8 markdown）
- [ ] 待用户跑 notebook，输出 EDA 数据

### Cell 清单

| Cell | 内容 |
|------|------|
| M ↓ C1 | 环境准备 + 路径 + set_seed(42) |
| M ↓ C2 | 解析所有 XML，统计每类图像数 + bbox 数 |
| M ↓ C3 | 每类 train/val 柱状图 |
| M ↓ C4 | 每类抽 1 张可视化（原图 + GT bbox） |
| M ↓ C5a | BBox 宽/高/面积/宽高比直方图 |
| C5b | COCO 三档（S/M/L）统计表 + < 30 告警 |
| C5c | 每类面积箱线图 + S/M/L 堆叠柱状图 |
| M ↓ C6 | 图像尺寸一致性检查 |
| M | EDA 结论摘要 |

### EDA 结果（2026-06-03 实测）

| 指标 | 数值 |
|------|------|
| 有效图像 | 1799 张（craize/crazing_240.jpg 缺标注，已排除） |
| 总 bbox | 4186 个 |
| 每图平均 bbox | 2.33 个 |
| 图像尺寸 | 全部 200×200 ✅ |
| bbox 宽 | min=8, median=55, max=199 |
| bbox 高 | min=9, median=77, max=199 |
| bbox 面积 | min=108, median=4704, max=39601 |
| bbox 宽高比 | median=0.68 |

#### COCO 三档分布

| 档位 | bbox 数 | 占比 |
|------|---------|------|
| Small (< 32²) | 447 | 10.7% |
| Medium (32²-96²) | 2772 | 66.2% |
| Large (≥ 96²) | 967 | 23.1% |

#### 每类细节

| 类 | train | val | bbox | small | medium | large |
|----|-------|-----|------|-------|--------|-------|
| crazing | 239 | 60 | 686 | **0** | 393 | 293 |
| inclusion | 240 | 60 | 1011 | 329 | 646 | 36 |
| patches | 240 | 60 | 881 | 56 | 684 | 141 |
| pitted_surface | 240 | 60 | 432 | **3** | 54 | 375 |
| rolled-in_scale | 240 | 60 | 628 | **16** | 502 | 110 |
| scratches | 240 | 60 | 548 | 43 | 493 | **12** |

#### 四个高危组合（< 30 bbox）

按 PROJECT_BRIEF §4.1 注 2：这四组 mAP 高方差，简历上不强报分尺度数字。

- crazing small: **0** — 该类的裂纹是网络状大面积缺陷，不存在小目标
- pitted_surface small: **3** — 凹坑几乎都是中大目标
- rolled-in_scale small: **16** — 氧化皮压入多为中等尺寸
- scratches large: **12** — 划痕是细长条，面积偏小但跨度大

#### 训练启示

- `crazing` 是 hardest class（网状、边界模糊、不含小目标）
- `inclusion` 有 329 个 small bbox（32.5%），mosaic 增强对提升 small recall 关键
- 200×200 → 640×640 放大 3.2×，small 目标在 feature map 上约 102×102 px，YOLOv8 stride=32 时可感知
- 宽高比 median=0.68，anchor 预设可能需要微调（默认 anchor 偏正方形）

### 待填

---

## Day 2.5 — 2026-06-03：固定数据集 split

### 完成事项

- [x] `src/split_dataset.py` 完成并跑通
- [x] `data/splits/{train,val,test}.txt` 已生成
- [x] `train.txt` 的 sha256 已记录，用于防篡改验证

### 分裂方案

- 方法：按类分层（stratified by class），seed=42
- 比例：80/10/10
- 文件列表均 `sorted()` 后再 split（保证幂等）

### 分裂结果

| Split | 行数 | crazing | 其他 5 类 |
|-------|------|---------|-----------|
| train.txt | 1439 | 239 | 240 |
| val.txt | 180 | 30 | 30 |
| test.txt | 180 | 30 | 30 |

### 锁仓

```
train.txt sha256 = 26dac896fb2e21bfd89550ebf419e5ca9388e1cdc455df9a51cdc2ab895c88fa
```

后续任何对 split 的误改动都能通过对比 sha256 发现。

### 关键设计决策

- 排除 craze/crazing_240.jpg（无标注），总计 1799 张有效图像
- `round(n * 0.8)` / `round(n * 0.1)` 取整，test 用减法兜底
- 输出文件内部 sorted，便于 diff
- 脚本幂等：重复运行 sha256 不变

---

---

## Day 3 — 2026-06-03：YOLOv8 训练跑通

### 完成事项

- [x] `src/convert_voc_to_yolo.py` — PASCAL VOC XML → YOLO TXT
- [x] `configs/data.yaml` — 指向 splits
- [x] `train.py` — 5 epoch 干跑 → 50 epoch 完整训练

### 训练结果

| 指标 | 值 |
|------|-----|
| 模型 | yolov8n |
| Epochs | 50 |
| 优化器 | AdamW (auto) |
| Best val mAP@0.5 | **0.774** (epoch 46) |
| Best val mAP@0.5:0.95 | **0.459** (epoch 48) |
| GPU 显存 | 2.3 GB / 6 GB |
| 训练耗时 | 28 min (~34 s/epoch) |

### 训练曲线特征

- 前 10 epoch mAP 快速上升（0.41 → 0.66），
- 10-30 epoch 缓慢爬坡（0.66 → 0.76），
- 30-50 epoch 波动横盘，best 出现在 epoch 46
- 说明 50 epoch 已接近收敛，需要换策略而非加轮数

### 已知问题

- workers=0（按计划先跑通，Day 4 测 2/4）
- 基线 mAP@0.5 = 0.774 vs 目标 0.85 → Day 4 调参追点
- 有一个 GitHub 下载超时问题（yolo26n.pt 对比模型），已加 `plots=False` 规避

### 环境修复记录

- pip install ultralytics 覆盖了 torch 2.5.1+cu121 → 2.12.0+cpu，手动恢复

---

## Day 4 — 2026-06-03：调参实验

### 实验汇总（统一 conf=0.001, iou=0.6 重新评测）

| Exp | 配置 | val mAP@0.5 | crazing mAP | Δ vs baseline |
|-----|------|-------------|------------|---------------|
| baseline | yolov8n, 50e, mosaic=1.0 | 0.782 | 0.289 | — |
| Exp 1 | yolov8s, 50e | 0.777 | 0.315 | -0.005 |
| Exp 2 | +cosine LR | 0.789 | 0.293 | +0.007 |
| Exp 3 | cosine + 100e | 0.781 | 0.305 | -0.001 |
| **Exp 4** | **mosaic=0 + 保守 aug + lr=0.005** | **0.800** | **0.430** | **+0.018** |

### 关键发现

1. **yolov8s 对小数据集无效**（Exp 1: 0.777 < baseline 0.782）
2. **Cosine LR 有效但提升有限**（Exp 2: +0.007）
3. **100 epoch 对 NEU-DET 无益**（Exp 3: 0.781），早期过拟合
4. **关 mosaic 是最高 ROI 改动**（Exp 4: crazing +0.14，整体 +0.018）
5. **mosaic 的效果是类别的**：对 patches/inclusion 有利但对 crazing 有害

### 评测协议修复

- 创建 `src/utils/eval_protocol.py`，固化 EVAL_CONFIG (conf=0.001, iou=0.6)
- 区分评测配置和生产配置 (conf=0.25)
- Day 4 所有实验用统一口径重新评测

### 复现性自检

| 运行 | mAP@0.5 | 浮动 |
|------|---------|------|
| Exp 4 原始 | 0.7997 | — |
| Exp 4 同 seed 重跑 | 0.7997 | **0.00000** |

✅ 通过。`set_seed(42) + deterministic=True` 组合在 NEU-DET 上完全可复现。

---

## Insight #1: 评测口径不统一 + mosaic 对纹理类有害

### 现象
Day 4 前 3 次实验整体 mAP 始终在 0.77-0.78，怎么调都不动。

### 根因（两层）
1. **整体 mAP 掩盖类间差距**：per-class 显示 5 类已 0.88+，crazing 只有 0.27
2. **mosaic 对 crazing 有毒**：mosaic 把 4 张 200×200 拼成一张训练图，crazing 的
   全网状纹理被切碎，模型 75% 时间看到的"crazing"都是不完整碎片

### 修复
1. Exp 4 关 mosaic + 降低 scale/HSV 增强 → crazing mAP: 0.289 → 0.430 (+49%)
2. 新建 eval_protocol.py 固化评测配置，区分评测/生产
3. 所有实验用 conf=0.001 统一重评

### 教训
1. mAP 整体值会骗人 — 必须每次看 per-class，特别是难类 Recall
2. 数据增强不是无脑开 — 要按缺陷类型选：区域型开 mosaic，纹理型关 mosaic
3. 调参 3 次不动就该 per-class 而不是继续动 epoch/LR
4. 撞到 0.80 的过程比 0.800 本身更值钱
5. **conf 阈值口径影响 mAP 0.01-0.02** — 比想象的大。所有评测必须显式传 conf
6. **val 和 test 是两个独立 split，不能互相替代**：Day 4 调参用 val，Day 5 终值用 test。
   如果用 val 数字写简历，是隐性的过拟合（test 0.745 vs val 0.800 就是警示）

---

---

## Day 5 — 2026-06-03：Test Set 终值 + Bad Case

### 任务 1: Test Set 终值

| Class | Test mAP@0.5 | Val mAP@0.5 | Δ |
|-------|-------------|------------|----|
| crazing | **0.291** | 0.430 | -0.139 |
| inclusion | **0.858** | 0.812 | +0.046 |
| patches | **0.943** | 0.993 | -0.050 |
| pitted_surface | **0.946** | 0.986 | -0.040 |
| rolled-in_scale | **0.578** | 0.598 | -0.020 |
| scratches | **0.855** | 0.980 | -0.125 |
| **Overall** | **0.745** | 0.800 | **-0.055** |

### 任务 2: 分尺度 BBox 分布

- Test set GT: 408 bboxes → S/M/L = 40/277/91 (9.8%/67.9%/22.3%)
- **18 个 cell 中 10 个 < 30 样本** → 不推荐报 per-scale mAP
- NEU-DET 本质是单一尺度数据集（67.9% medium）

### 任务 3: Bad Case 分析

9 个最难样本：

| # | 类 | 错误类型 | 根因 |
|---|-----|---------|------|
| 1-3 | crazing | 漏检 | 低对比度，网状纹理无明确边界 |
| 4-6 | crazing | 误检 | 纹理噪声被误判（crazing 类区分度低） |
| 7 | patches | 低 IoU | 形状不规则，预测框不贴合 |
| 8-9 | inclusion | 低 IoU | 边界模糊，框覆盖不全 |

### Test set 终值 = 0.745，简历数字来源

```
test set mAP@0.5 = 0.745 (180 images, conf=0.001 COCO standard)
```

低于 val 0.800 (-0.055)。主要原因：
1. crazing val 0.430 → test 0.291 (0/0 的 small bbox 训练数据)
2. scratches val 0.980 → test 0.855 (样本选择偏差)

这不是模型崩了，是 val/test split 分布差异的真实反映。Day 2.5 的分层
80/10/10 保证了类均匀，但未保证难度均匀。

### Week 1 整体收尾

```
- 5 天完成：EDA → split → 训练 → 5 次实验 → test 终值 + bad case
- val mAP@0.5 = 0.800 / test mAP@0.5 = 0.745
- 关键发现：mosaic 对 crazing 有毒（+49% to 0.430），但 val→test 迁移差
- 复现性：同 seed 重跑 mAP 浮动 0.00000
- 下一个瓶颈：crazing 基础表征能力（Week 2 U-Net 分割或提供像素级边界）
```

---

## Insight #2: val 0.800 vs test 0.745 的 -0.055 差距溯源

### 现象
Exp 4 best.pt：
- val (180 张) mAP@0.5 = 0.800
- test (180 张) mAP@0.5 = 0.745
- 差距 -0.055，远超复现性浮动 0.000

### 诊断过程
sha256 验证：train.txt = 26dac896fb...，与 Day 2.5 一致，split 未被改动。

**逐类拆解 mAP 差距**：

| 类 | val | test | Δ |
|---|---|---|---|
| crazing | 0.430 | 0.291 | **-0.139** |
| scratches | 0.969 | 0.855 | **-0.114** |
| patches | 0.993 | 0.943 | -0.050 |
| inclusion | 0.812 | 0.858 | +0.046 |
| pitted_surface | 0.982 | 0.946 | -0.036 |
| rolled-in_scale | 0.598 | 0.578 | -0.020 |

scratches + crazing 贡献了 -0.042 / -0.055 = 76% 的整体差距，其余 4 类合计 -0.013。

### 根因 A：scratches bbox 面积分布偏差（可证伪、可修复）

| split | bbox 数 | 面积均值 | 中位数 |
|---|---|---|---|
| train | 454 | 3683 | 3113 |
| val | 40 | 4882 | 4433 |
| test | 54 | **2582** | **2226** |

test 的 scratches 平均面积只有 val 的 53%。YOLOv8 对小目标天生劣势，
面积 -47% 直接转化为 mAP -11.4%。

**根因**：split_dataset.py 做的是 stratified by class（按类分层），
但没有 stratified by bbox size。scratches 是细长缺陷，类内尺寸方差大，
test 偶然抽到了较小的样本子集。

### 根因 B：crazing 是小样本统计方差（可证、不可修）

| 指标 | val (59 bbox) | test (67 bbox) | Δ |
|---|---|---|---|
| bbox 面积均值 | 10343 | 10266 | -0.7% |
| bbox 面积中位数 | 9620 | 9460 | -1.7% |
| 灰度均值 | 138.1 | 128.9 | -6.7% |
| 灰度方差 | 994.1 | 914.5 | -8.0% |
| Sobel 梯度均值 | 87.1 | 81.2 | -6.8% |

bbox 大小、纹理对比度、灰度统计都几乎一致（差 < 8%），
crazing 暴跌 -0.139 无法用任何可观察变量解释。

**根因**：每类 30 张测试图的 AP 估计标准差约 0.05-0.10，
crazing 0.139 的偏离在 2σ 范围内，是 n=30 的统计学必然。
K-fold 交叉验证能平滑掉此类方差，但 4 周项目周期不做。

### 教训
1. **stratified split 必须双维度**：按类 + 按 bbox 尺寸分位数（如四分位）
   双重分层，避免类内尺寸偏差
2. **每类 30 张的 test set 不够稳定**：AP 估计标准差 0.05-0.10，
   小数据集做 portfolio 项目时应该考虑 K-fold CV
3. **不要只看整体 mAP**：拆到 per-class + 拆到 bbox 尺度，才能定位
   差距的真实来源
4. **未达指标 ≠ 项目失败**：能解释清楚 mAP 差距来源比假装达标更有价值。
   面试中"我知道为什么我的数字不到 0.80"远胜于"我达到了 0.80"

### 不做重训的决策
重新 split + 重训 5 个实验需要 2-3 小时，但：
- crazing 偏差是统计学的，重 split 后会出现在别的类上
- scratches 偏差是 split 策略问题，需要改 split_dataset.py
- 当前数字虽未达最低线 0.80，但可解释、可写进简历

决策：**接受 test mAP@0.5 = 0.745 作为终值**，把这次诊断作为
"撞到小数据集天花板时如何分析"的 STAR 故事进简历。

---

## Insight #3: 分尺度 mAP 评测的样本数警戒线

### 现象
按 v1.1.1 BRIEF §4.1 注 2 要求，做 COCO 口径分尺度 mAP（原图 200×200 坐标系）。
结果三档样本分布严重不均：

| 尺度 | bbox 阈值 | 数量 | 占比 |
|---|---|---|---|
| small  | < 32² = 1024     | 40  | 9.8%  |
| medium | 32² - 96² = 9216 | 277 | 67.9% |
| large  | ≥ 96²            | 91  | 22.3% |

进一步拆到 6 类 × 3 尺度的 18 个 cell：
- **10 / 18 cell 样本数 < 30**，AP 估计极不稳定
- 例如：crazing × small = 0 个 bbox，无法评测

### 决策
按 v1.1.1 BRIEF §4.1 注 2 的"≥ 30 警戒线"规则：
- 仅报整体三档 mAP（small/medium/large 各档至少 40 个 bbox）
- per-class × per-scale 的 18 个数字**不进简历**，仅作 DEVLOG 留档
- README 验收表里"分尺度 mAP" 三行仍然写，但备注"基于 n=40/277/91"

### 教训
1. **NEU-DET 原图 200×200 的设计下，small 档天然稀缺**：
   缺陷尺寸普遍占图像 > 5%，不像 COCO 那种大图里挑小目标
2. **分尺度分析对工业缺陷数据集要谨慎用 COCO 口径**：
   建议用数据集自身 bbox 分布的四分位数划档（数据驱动）
3. **每个 cell ≥ 30 样本是统计稳定的最低线**，不到就声明"仅参考"

---

---

## Insight #4: U-Net 弱监督分割的两次踩坑 + 多类训练负迁移发现

### 现象链

**阶段 1：自动生成"人工 mask"**
- CC 在 Day 5.5 跳过人工标注，用 bbox→polygon 脚本自动生成 30 张评测 GT
- 评测 baseline U-Net IoU = 0.72, refined v1 IoU = 0.02
- 两个数字都不可用（baseline 是循环论证，refined 是 mask 形状不匹配）

**阶段 2：人工 30 张 polygon mask + 重评 baseline**
- LabelMe 手工标 30 张真 polygon mask
- 评测 baseline U-Net Macro IoU = 0.413 (vs 30 真 polygon)
- per-class: pitted=0.748, crazing=0.524, patches=0.495,
  rolled-in_scale=0.276, scratches=0.272, inclusion=0.162

**阶段 3：refined U-Net v1 全零退化**
- vs 30 真 polygon: Macro IoU = 0.005
- 4 步诊断：mask 稀疏 / 模型权重 / 评测口径 / 训练参数
- 真因：OTSU_BIAS=+0.15 让 3 类 refined mask 前景 < 1%
  - inclusion fg = 0.0% (完全黑)
  - pitted_surface fg = 0.1%
  - rolled-in_scale fg = 0.2%
- 模型正确学到了"99.9% 背景"的退化解

**阶段 4：refined U-Net v2 重训**
- 改 OTSU_BIAS = -0.10
- 6 类前景占比进入合理区间 (13-75%)
- 人眼校验 6 张可视化图（PLAN P-6 Step 2 强制步骤）
- 训练 50 epoch，best manual IoU = 0.366 at epoch 5
- 训练曲线显示分布偏移：train_loss 持续下降 (0.256 → 0.034)
  但 manual IoU 在 epoch 5 后稳定在 0.34，
  证明 refined mask 与真 polygon mask 之间存在约 0.37 的代表性上限

### baseline vs refined v2 per-class 对比

| class | baseline | refined v2 | Δ | 解读 |
|---|---|---|---|---|
| pitted_surface | 0.748 | 0.802 | +0.053 ✅ | Otsu refine 起作用 |
| rolled-in_scale | 0.276 | 0.319 | +0.043 ✅ | Otsu refine 起作用 |
| scratches | 0.272 | 0.325 | +0.053 ✅ | Otsu refine 起作用 |
| crazing | 0.524 | 0.518 | -0.007 ≈ | 持平 |
| inclusion | 0.162 | 0.157 | -0.005 ≈ | 持平（refined mask 实际退化为 bbox 矩形） |
| patches | 0.495 | 0.073 | **-0.422 🔴** | 关键发现：多类训练负迁移 |
| Macro | 0.413 | 0.366 | -0.047 | refined 整体略低于 baseline |
| Micro | 0.472 | 0.469 | -0.003 ≈ | 持平 |

### 关键发现：patches 暴跌的真因

可视化显示 patches 的 refined mask **实际还是 bbox 矩形**
（Otsu 在 patches bbox 内没找到比阈值更暗的像素，fallback 到全填充）。
按理说 refined U-Net 应该和 baseline 学到一样的 patches pattern。

但实测 patches IoU 从 0.495 → 0.073，暴跌 -0.42。

**真因：弱监督多类分割的负迁移**
- baseline 训练：6 类全是矩形伪 mask，U-Net 学单一 pattern
- refined v2 训练：3 类是不规则 mask（crazing/pitted/scratches），
  3 类是矩形 mask（inclusion/patches/rolled-in_scale）
- U-Net 容量有限（resnet34，~24M params），mixed-mask 训练下，
  模型重新分配学习能力到新 pattern，patches 这种简单 pattern 被遗忘

### 教训
1. **AI 自动生成的"人工标注"是循环论证**——评测 GT 必须独立于训练标签
2. **mask 质量必须人眼校验**——统计学信号不够（fg 占比对了不代表位置对）
3. **退化解（全零 / 全一）是弱监督的常见 trap**——必须检查 raw logit 范围，
   不能只看 IoU
4. **OTSU_BIAS 是类别敏感的**——高对比度类能 refine，
   低对比度类退化到 bbox 矩形
5. **弱监督多类训练有标签一致性要求**——若不同类用不同标签生成策略
   （矩形 + 不规则混合），简单类会因负迁移退化
6. **训练曲线 vs 评测曲线的分布偏移**是弱监督本质上限的体现，
   不是训练 bug

### Week 2 最终决策
Best U-Net = baseline_best.pt (Macro IoU = 0.413)
保留 refined_v2_best.pt 作为对照实验数据
DEVLOG 详细记录两次踩坑 + 关键发现，作为面试讨论素材

---

## Insight #5: Python 环境管理踩坑 + 迁移到独立 env

### 现象链

**阶段 1: Week 1-2 训练（错误状态下完成）**
- 项目按 PROJECT_PLAN 设计应该跑在 conda env "pytorch"
- 实际所有训练（YOLOv8 5 次 ablation + U-Net baseline/refined）跑在 base env
- 原因：conda 默认 auto_activate_base=true，开终端自动进 base，未显式 mamba activate pytorch
- 结果：Week 1-2 全部在 base 完成，跑得通但环境"不该是这样"

**阶段 2: Day 11 PyQt6 装包冲突**
- pip install PyQt6==6.6.1 装到 base env
- 启动报 "DLL load failed while importing QtCore"
- 4 步诊断：
  1. VC++ Redistributable → 系统已装 2015-2022 完整版，排除
  2. PyQt6 安装文件 → DLL 都在 site-packages/PyQt6/Qt6/bin/，完整
  3. PATH 冲突 → 没有 Qt 相关路径
  4. **真因**：base env 同时存在 conda 装的 PyQt5 (5.15.11) 和 pip 装的 PyQt6 (6.6.1)
     Windows DLL 搜索顺序：conda Library\bin\ 在 PATH 前面 →
     Python import PyQt6 时优先加载到 Qt5Core.dll → 二进制不兼容 → ImportError

**阶段 3: 追溯 PyQt5 来源**
- conda 装的 spyder 6.0.5 硬依赖 pyqt 5.15.11
- 装 spyder 时 conda 顺带把 qt-main / qt-webengine / pyqtwebengine 一起装进 base
- 这是"误装"的根源 — 项目并不需要 spyder

**阶段 4: 修复策略选择**
- 选项 A：给 base 加 DLL 路径补丁（os.add_dll_directory）→ 治标
- 选项 B：迁移所有项目操作到独立 pytorch env → 治本
- 决策：选 B。理由：环境隔离比 DLL 补丁更工程化，未来打包 .exe 也方便

**阶段 5: 执行迁移 + 清理**
1. pytorch env 装齐 PyQt6 四件套（6.6.1 + Qt6 6.6.3 + sip 13.11.1 + Fluent 1.5.7）
   - CC 自动把 PyQt6-Qt6 从 6.11.1 降到 6.6.3，发现是跨大版本不兼容问题
   - 锁定 4 个版本组合到 requirements.txt
2. 弹窗测试通过：QMainWindow + QApplication 能正常显示和退出
3. 验证 Week 1 best.pt 在 pytorch env 加载 OK
4. 备份 pytorch env：mamba env export → pytorch_env_snapshot.yml
5. 清理 base：
   - pip 卸载：torch / ultralytics / PyQt6 全家 / labelme / onnxruntime /
     segmentation_models_pytorch / torchvision（-100 包）
   - conda 卸载：spyder / spyder-kernels / qtconsole / pyqt / qt-main /
     qt-webengine / pyqtwebengine / pyqt5-sip（-120 包）
6. 关闭 base 自动激活：conda config --set auto_activate_base false

### 最终状态

| 环境 | 用途 | 包数 |
|---|---|---|
| base | conda 自身 + 基础工具 | 350 个（清理前 790） |
| pytorch | 所有项目操作 | ~200 个 |

未来工作流：
- 开终端不再自动进 base
- 项目操作必须先 mamba activate pytorch
- 项目根目录有 activate_env.bat 一键激活

### 教训

1. **conda auto_activate_base 默认 true 是个坑**
   新装 miniforge 后第一件事应该是 conda config --set auto_activate_base false。
   否则会"以为在自己的 env，其实在 base"，环境隔离名存实亡。

2. **Windows 上 conda + pip 混用要格外小心**
   pip 不知道 conda 装了什么，conda 不知道 pip 装了什么。
   特别是 GUI 框架（Qt5 / Qt6）跨大版本时，DLL 加载顺序会冲突。

3. **CC（AI 编程助手）在依赖诊断上容易归因偏差**
   第一次报错 CC 归因到"VC++ Redistributable 缺失"（错的）。
   排查后才发现真因是 Qt5/Qt6 共存。
   教训：AI 给的"看起来合理"的修复方案要先验证再执行。

4. **迁移环境而非打补丁**是更工程化的选择
   os.add_dll_directory 能让 PyQt6 跑起来，但环境永远是乱的。
   花 30 分钟正规迁移到独立 env，比之后每次踩坑修补丁省时间。

5. **环境清理前必须备份**
   清理 base 前导出 pytorch_env_snapshot.yml + git commit。
   万一清理误伤，能从备份恢复。这次没用上，但下次可能就用上了。

6. **base env 应该只放 conda 自身**
   - ✅ 应该在 base：conda、pip、setuptools、wheel
   - ❌ 不该在 base：torch、numpy、应用类库、IDE
   - 一旦 base 装了应用包，迟早出现环境混乱

### 简历可写素材

"项目初期 conda env 未严格隔离，Week 1-2 训练实际跑在 base，
与 system IDE 的 PyQt5 产生 DLL 冲突阻塞 Week 3 GUI 开发。
诊断后采用'迁移而非补丁'策略：将所有项目操作正规化到独立 pytorch env，
base 清理回归 conda 自身（-220 个包）。教训：Python 项目从 day 1 必须
显式激活独立 env，并禁用 base 自动激活。"

---

## Day 11: PyQt6 SCADA 风主窗口骨架完工 — 2026-06-04

### 完成内容
- 13 个新文件（9 个有内容 + 4 个 __init__.py），总代码量约 800 行
- 三层架构创建：src/ui (5 文件) + src/manager (3 占位) + src/algo (空目录) + config + utils
- 主窗口尺寸 1280×800，三栏布局 + 顶部菜单 + 底部事件日志 + 状态栏
- 所有 widget 用纯 PyQt6 + 手写 QSS 实现（未用 qfluentwidgets，避免 1.5.7 API 不确定性）

### 环境确认
- pytorch env 全齐：torch 2.5.1+cu121 + PyQt6 6.6.1 + ultralytics 8.4.60 + labelme 5.5.0
- `mamba activate pytorch` → `python main.py` 一键启动
- 用户工作流验证：7/7 验收项全部 PASS

### 关键设计决策
1. 用 moveToThread 模式而非继承 QThread（Day 12 实现 Worker 时按此规范）
2. Manager 三个类先空壳，signal 声明完整，方法体 pass（Day 12 才填实现）
3. 所有颜色 / 字体 / 尺寸常量集中到 theme.py + app_config.py，禁止硬编码
4. 双输出 logger（终端 + UI 事件面板），统一 logging 入口
5. 选原生 PyQt6 + QSS 而非 qfluentwidgets，避免依赖锁版本踩坑

### 工程化亮点（简历可用）
- 严格三层架构隔离：UI 不调算法、Manager 不写 widget、Algorithm 无 Qt 依赖
- 多线程规范文档化（待 Day 12 验证）：UI 主线程，算法 Worker 线程，signal/slot 通信
- 用户工作流验证：activate_env.bat → python main.py 一键启动

### 待 Day 12 验证
- Worker 模板（moveToThread + 5 步 connect）
- 取消机制（_cancel_flag）
- 异常 emit 不静默

## Day 12: 单图检测 + 多线程 Worker 实战 — 2026-06-04

### 完成内容
- src/algo/ 完整实现：yolo_detector.py (85 行) + postprocess.py (105 行)
- src/manager/ 完整实现：_worker.py (90 行) + inference_manager.py (157 行)
- src/ui/ 增强：image_viewer.py (+80) + main_window.py (+105) + control_panel.py (+57)
- 端到端单图检测功能 + 多线程 Worker 模式落地

### 关键工程发现

#### Insight #6: ultralytics 8.x CUDA 不自动绑定
- 初次实现 Day 12 时推理跑在 CPU（276ms / 200×200 图）
- 排查发现 ultralytics 8.x 版本 model 默认加载到 CPU，predict() 默认也用 CPU
- 必须显式 `model.to('cuda')` + `predict(device='cuda')` 才能用 GPU
- 修复后稳态推理从 276ms → 21.5ms（提速 9.5×）
- 注意：CUDA warmup 含 kernel 编译，cold start ~1073ms（一次性，可忽略）

#### Insight #7: crazing 类置信度天然偏低，conf=0.25 默认值过滤误删
- 用 conf=0.25 时 crazing_1.jpg 检测 0 个目标
- 排查发现 crazing 类整体 top 置信度仅 0.228（Week 1 训练数据决定）
- 根因：crazing 是 NEU-DET 中纹理弥散类，没有清晰边界，YOLO bbox 难拟合
- 修复：conf_threshold 默认 0.25 → 0.10
- 教训：通用 conf 默认值不适合 fine-grained 缺陷分类，应按类别分别设阈值

### 多线程规范落地验证（Day 11 立的规则全部生效）
1. ✅ Worker 继承 QObject 而非 QThread
2. ✅ moveToThread 模式正确实施
3. ✅ 5 步 connect 套路：started→run, finished→on_finished, error→on_error,
   finished+error→quit, finished→cleanup
4. ✅ try/except 包裹 run()，异常通过 error signal 发出（不静默）
5. ✅ _cancel_flag 取消机制（用户验证 PASS）
6. ✅ thread.quit() + wait() + deleteLater() 完整生命周期
7. ✅ Algorithm 层零 Qt 依赖（grep "import PyQt" src/algo/ 空结果）

### 简历可用素材
- "采用三层架构（UI/Manager/Algorithm）+ QThread + signal/slot 多线程模式，
  GPU 推理稳态 21.5ms，主线程零卡顿。排查 ultralytics 8.x CUDA 不自动绑定坑，
  GPU 启用后推理提速 9.5×。"
- "针对 NEU-DET crazing 纹理类置信度天然偏低（top 0.228），将默认 conf 阈值从
  0.25 调整为 0.10，确保所有 6 类缺陷都能正常检测。"

### Day 13 待处理
- 按类别分别配置 conf 阈值（per-class threshold，crazing 用 0.05，其他 0.20）
- 批处理 Worker（不止单图，整个文件夹批量推理）
- GPU 监控显示（device、显存使用、加到状态栏）
- 图像列表按类别分组（当前按字母排序）

---

## Day 13: 批处理 Worker + per-class threshold + 全量验证 — 2026-06-04

### 完成内容
- src/manager/_batch_worker.py (151 行) - BatchWorker 含取消 + EMIT_EVERY=10
- src/manager/_csv_writer.py (74 行) - 零 Qt 依赖的 CSV 写入工具
- src/manager/batch_manager.py 重写 (204 行) - 5 步 connect + 文件夹扫描
- src/algo/yolo_detector.py + postprocess.py - per-class threshold 后处理
- src/ui/main_window.py + control_panel.py - 进度条 + 文件夹对话框 + 批处理事件
- 全量验收：1800 张 33.7s（17.3ms/img），取消延迟 19ms

### 实测数据（NEU-DET 完整数据集）
- 1800 张总耗时：33.7s（理论下限 31.1s，开销 < 9%）
- 平均推理：17.3ms/img（比 Day 12 单图 21.5ms 快——批处理 warmup 摊销）
- 进度 emit：180 次（严格 EMIT_EVERY=10）
- 缺陷检出率：1798/1800 (99.9%，含 0.05 crazing 阈值)
- 取消延迟：19ms（教科书级 Qt 多线程响应）
- 每类 300 张精确（rglob 递归扫描正确）

### Insight #8: rglob 递归扫描 + MAX_BATCH_FILES 保护
- 初版实现 rglob 递归扫描所有子目录（适应 NEU-DET 三级嵌套结构）
- 风险：用户选 D:\ 会遍历整个文件系统，导致主线程卡顿数分钟
- 修复：加 MAX_BATCH_FILES=5000 保护 + 空文件夹弹窗
- 教训：批处理工具必须有"防呆"边界保护，不能信任用户输入

### Insight #9: 取消时写部分 CSV
- 初版实现：取消即丢弃，154 张已跑结果不写入
- 修复：取消时写带 _cancelled 后缀的 CSV，保留已处理数据
- 工程原则：用户耗时投入（GPU 推理）不能因取消而归零

### 多线程规范深化验证（在长循环场景下）
1. ✅ _cancel_flag 在每张图开头检查（取消延迟 19ms 是关键验证）
2. ✅ EMIT_EVERY=10 平衡进度反馈与 signal 风暴（180 次而非 1800 次）
3. ✅ Worker 线程内写 CSV（不阻塞主线程）
4. ✅ cancelled signal 与 error signal 分离（取消不是异常）
5. ✅ 1800 张长循环无 GPU 显存泄漏（thread.deleteLater 正确生效）

### 简历金句
- "实现 NEU-DET 1800 张批处理工具，GPU 端到端耗时 33.7s（17.3ms/img），
  主线程零卡顿、取消延迟 19ms。设计 EMIT_EVERY=10 减少跨线程 signal 风暴 10×。
  per-class conf threshold 解决 fine-grained 类别置信度分布差异。"

### Day 14 待处理
- 报表生成（HTML/PDF 报表 from CSV）
- CSV 后处理可视化（matplotlib 类别分布柱状图）
- 加更精细的 GPU 监控显示
- 用户可选导出已处理图像的检测框图

---

## 日志格式模板

```markdown
## B??: [bug 简述]

- **现象**：[完整报错 / 异常行为]
- **触发命令**：[精确可复现的命令]
- **根因**：[为什么发生]
- **修复**：[改了什么]
- **教训**：[下次怎么避免]
```

---

## Day 13: 批处理 Worker + per-class threshold + 全量验证 — 2026-06-04

### 完成内容
- src/manager/_batch_worker.py (151 行) - BatchWorker 含取消 + EMIT_EVERY=10
- src/manager/_csv_writer.py (74 行) - 零 Qt 依赖的 CSV 写入工具
- src/manager/batch_manager.py 重写 (204 行) - 5 步 connect + 文件夹扫描 + 安全检查
- src/algo/yolo_detector.py + postprocess.py - per-class threshold 后处理
- src/ui/main_window.py + control_panel.py - 进度条 + 文件夹对话框 + 批处理事件
- 全量验证：1800 张 33.2s（17.1ms/img），取消延迟 18ms

### 实测数据（NEU-DET 完整数据集 1800 张）
- 总耗时 33.2s（理论下限 30.8s，开销 8%）
- 平均推理 17.1ms/img（比 Day 12 单图 21.5ms 还快——warmup 摊销）
- 进度 emit 180 次（严格 EMIT_EVERY=10）
- 缺陷检出率 1798/1800 (99.9%)
- 取消延迟 18ms（11/11 验证稳定）
- 每类 300 张精确（rglob 递归扫描正确）

### Insight #8: rglob 递归扫描 + MAX_BATCH_FILES 防呆保护
- NEU-DET 三级嵌套目录结构（train/images/{class}/），glob 找不到子目录文件
- 初版实现 rglob 递归扫描后未加边界保护，风险：用户选 D:\ 会扫整盘
- 修复：加 MAX_BATCH_FILES=5000 上限 + 空文件夹弹 info 对话框
- 实测：MAX_BATCH_FILES=100 + 1800 张文件夹 → 正确弹 warning + Yes/No 处理 OK
- 教训：批处理工具必须有"防呆"边界，不信任用户输入

### Insight #9: 取消时写部分 CSV（用户耗时不归零）
- 初版实现：取消即丢弃，已处理结果不写
- 修复：取消时写带 _cancelled 后缀的 CSV，CSV header + 已处理行
- 实测：327 张处理后取消 → 328 行 CSV，0 截断
- 工程原则：GPU 推理是用户投入的真实资源，取消不应归零

### 多线程规范深化验证（长循环场景）
1. ✅ _cancel_flag 每张图开头检查（取消延迟 18ms 验证）
2. ✅ EMIT_EVERY=10 平衡进度反馈与 signal 风暴（180 次 vs 1800 次，省 10×）
3. ✅ CSV 写入在 Worker 线程（不阻塞主线程）
4. ✅ cancelled signal 与 error signal 分离（取消不是异常）
5. ✅ 1800 张长循环无 GPU 显存泄漏（thread.deleteLater 正确）

### 简历金句
"实现 NEU-DET 1800 张批处理工具：GPU 端到端 33.2s（17.1ms/img），主线程零卡顿，
取消延迟 18ms。设计 EMIT_EVERY=10 减少跨线程 signal 风暴 10×。
per-class conf threshold 解决细粒度类别置信度分布差异。"

---
