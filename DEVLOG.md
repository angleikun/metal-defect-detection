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
