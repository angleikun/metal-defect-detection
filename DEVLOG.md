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
