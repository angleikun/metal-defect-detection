# 项目素材清单

## 1. 项目一句话定位

金属表面缺陷检测系统（NEU-DET 6 类 / YOLOv8 + U-Net / PyQt6 桌面应用 / 工业 SCADA 风）

## 2. 关键数字（按类别）

### 训练相关
- YOLOv8n test set mAP@0.5: **0.745** (Day 4 终值, conf=0.001 / iou=0.6 EVAL_CONFIG)
- YOLOv8n val set mAP@0.5: **0.800** (best epoch)
- YOLOv8 5 次 ablation 实验 (Exp 1-4 + baseline)
- 最佳配置: mosaic=0.0, lr0=0.005, cos_lr=True, epochs=50, imgsz=640, batch=16
- crazing class val mAP: 0.291 → test: further drop (small sample + texture variance)
- U-Net ResNet34 backbone Macro IoU (vs 30 human masks): **0.413** (baseline mask training)
- U-Net refined mask v2 best IoU: **0.366** (negative transfer from multi-class mixed-mask training)
- 30 张人工 polygon mask 标注 (LabelMe, Day 5.5)
- bbox→Otsu→morphology 弱监督 mask 生成管线 (OTSU_BIAS=-0.10 optimized)
- 数据集: 1800 张 NEU-DET (200×200 grayscale), 6 classes × 300 each
- 80/10/10 stratified split with sha256 locking

### 推理性能
- PyTorch GPU 单图 (warmup): 21.5ms (Day 12)
- PyTorch GPU 批处理 avg: **17.1ms/img** (Day 13), P50: 16.5ms
- PyTorch GPU FPS: **51.1** (Day 15 benchmark, 90 measured images)
- ONNX CPU avg: **39.9ms/img** (Day 15), P50: 39.7ms
- ONNX CPU FPS: **25.1** (Day 15)
- ONNX 一致性验证: 坐标误差 **< 0.06 px** (5/5 框数精确匹配)
- CUDA warmup (cold start): ~1073ms (一次性 kernel 编译)
- GPU/CPU ratio: **2.0×** (远低于常见的 10-20×)

### 工程指标
- 1800 张全量批处理: **33.2s** (Day 13)
- 1800 张批处理 overhead: < 9% (理论下限 30.8s)
- 取消延迟: **18ms** (Day 13, < 1 张图处理时间)
- 批处理进度 emit: 180 次 (EMIT_EVERY=10, 1800/10)
- 缺陷检出率 (per-class threshold): 1798/1800 (99.9%)
- HTML 报表生成 (1800 张 CSV): **0.7s** (Day 14)
- HTML 文件大小: **105.5 KB** (base64 内嵌 3 图表)
- ONNX 模型大小: **11.7 MB** (PyTorch 6.0 MB → 1.96×)
- 单图检测 GUI 启动: < 5s (含模型延迟加载)

### 代码规模
- 总 Python 代码量: **4,337 行** (src/ 4169 + 根目录 168)
- Python 文件数: **37 个** (src/ 34 + 根目录 3)
- git commits: **10 个** (Day 1 - Day 15)
- DEVLOG Insights: **12 条**
- 第 1 条 commit: `d966d61`
- 最新 commit: `2ab4070`

## 3. 全部 Insight 列表（按编号）

### Insight #1: 评测口径不统一 + mosaic 对纹理类有害
mAP 评测 conf 阈值不统一会导致 0.01-0.02 偏差；mosaic 增强对 crazing/patches 等纹理类造成 49% mAP 下降（纹理被拼接破坏）。

### Insight #2: val 0.800 vs test 0.745 的 -0.055 差距溯源
val/test gap 根因是 scratches 类 bbox 面积 test 仅为 val 的 53%（-47%），属于 split 统计偏差而非过拟合；crazing 类 gap 是 n=30 小样本统计方差。

### Insight #3: 分尺度 mAP 评测的样本数警戒线
NEU-DET small 档仅 447 个 bbox（10.7%），4 类不足 30 个；分尺度 mAP 在样本 < 30 时方差极大，仅作参考不强报简历。

### Insight #4: U-Net 弱监督分割的两次踩坑 + 多类训练负迁移发现
(1) OTSU_BIAS=0.15 导致 3 类前景 < 1% → 模型输出全零；(2) 改为 -0.10 后单类训练有效但多类混合训练产生负迁移（3 类不规则 + 3 类矩形特征互斥）。

### Insight #5: Python 环境管理踩坑 + 迁移到独立 env
base env 含 spyder PyQt5/Qt5 与 pip PyQt6 产生 DLL 加载冲突；诊断后采用"迁移而非补丁"策略，将所有项目 ops 正规化到独立 pytorch env。

### Insight #6: ultralytics 8.x CUDA 不自动绑定
ultralytics 8.x 加载模型默认到 CPU，predict() 默认也用 CPU；必须显式 `model.to('cuda')` + `predict(device='cuda')` 才能用 GPU；修复后推理从 276ms → 21.5ms（提速 9.5×）。

### Insight #7: crazing 类置信度天然偏低，conf=0.25 默认值过滤误删
crazing 类 top 置信度仅 0.228，conf=0.25 直接漏检 0 boxes；根因是 crazing 纹理弥散无清晰边界 YOLO bbox 难拟合；修复为 conf=0.10 全局阈值 + Day 13 per-class threshold。

### Insight #8: rglob 递归扫描 + MAX_BATCH_FILES 防呆保护
NEU-DET 三级嵌套目录结构 glob 找不到子目录文件；rglob 递归扫描未加边界保护 → 用户选 D:\ 会扫整盘；修复加 MAX_BATCH_FILES=5000 上限 + 空文件夹弹 info 对话框。

### Insight #9: 取消时写部分 CSV（用户耗时不归零）
初版取消即丢弃已处理结果；修复为取消时写带 _cancelled 后缀的 CSV，保留已处理数据；工程原则：GPU 推理是用户投入的真实资源，取消不应归零。

### Insight #10: HTML 报表的工程决策
(1) base64 内嵌 vs 独立 PNG：选 base64 单文件可分发给客户/打印/邮件；(2) 浅色风 vs SCADA 暗色：HTML 选白底打印友好，GUI 保留 SCADA 暗色操作员盯屏；(3) matplotlib Agg 后端必须在 import pyplot 之前，否则 Qt 冲突。

### Insight #11: ONNX CPU 推理的工业意义
CPU 25 FPS @ 200×200 已覆盖典型产线节拍 (1-10 张/秒)；GPU/CPU 仅 2× 差距（远低于常见 10-20×）因为选 YOLOv8n + 200×200 小图对 CPU 友好；模型选型应考虑部署目标。

### Insight #12: 一致性验证是工程必需
很多 ONNX 转换跳过一致性测试导致部署后发现精度退化才返工；0.06 px 误差证明转换无损；工程原则：模型格式转换必须做"同输入同输出"验证。

## 4. 技术栈（按层级）

### 深度学习训练
- PyTorch 2.5.1 + CUDA 12.1
- ultralytics 8.4.60 (YOLOv8n)
- segmentation_models_pytorch 0.3.4 (U-Net ResNet34)
- torchvision 0.20.1
- numpy 2.2.6 / pandas 2.3.3
- pycocotools 2.0.7 (COCO mAP 评测)

### GUI 框架
- PyQt6 6.6.1 + Qt6 6.6.3
- PyQt6-sip 13.11.1
- PyQt6-Fluent-Widgets 1.5.7 (最终未使用，改用原生 widget + QSS)
- 原生 PyQt6 QWidget + 手写 QSS SCADA 暗色主题
- QThread + moveToThread 多线程模式
- signal/slot 5 步 connect 套路

### 图像处理 / 可视化
- OpenCV 4.13 (cv2.dnn.NMSBoxes, letterbox, BGR/RGB)
- matplotlib 3.10.9 (Agg backend, base64 内嵌 HTML)
- LabelMe 5.5.0 (polygon 标注)

### 推理部署
- ONNX Runtime 1.23.2 (CPUExecutionProvider)
- onnx 1.21.0 (opset=12, simplify)
- onnxslim 0.1.94

### 报表 / 导出
- openpyxl 3.1.5 / reportlab 4.2.2 (Week 3 预装)
- HTML + base64 单文件报表 (f-string 模板, 无外部依赖)

### 环境管理
- miniforge3 / mamba
- pytorch env 独立隔离
- 四件套版本锁定: PyQt6=6.6.1 + Qt6=6.6.3 + sip=13.11.1 + Fluent=1.5.7

## 5. 工程化亮点（关键词清单）

- 三层架构: UI / Manager / Algorithm (零 Qt 依赖的 Algorithm 层)
- 多线程 Worker: moveToThread 模式 (不继承 QThread)
- 5 步 connect 套路: started→run, finished→on_finished, error→on_error, finished+error→quit, finished→cleanup
- per-class conf threshold: crazing 0.05 / inclusion 0.20 / patches 0.20 / pitted 0.15 / rolled 0.10 / scratches 0.20
- 取消机制: _cancel_flag + 18ms 延迟 + _cancelled CSV
- EMIT_EVERY=10: signal 风暴控制, 1800 张仅 180 次 emit (省 10×)
- 环境隔离: pytorch env 迁移, base 清理 -220 个包, auto_activate_base=False
- DLL 加载顺序排查: Qt5 in conda Library\bin\ precedence → 迁移 env 解决
- CSV 自包含报表: base64 内嵌 3 图表, 单文件 105.5KB, 浅色打印友好
- ONNX 一致性验证: 5/5 框数匹配, 坐标误差 < 0.06 px
- rglob 递归扫描 + MAX_BATCH_FILES=5000 防呆保护
- 实验可复现性: set_seed(42) + cudnn.deterministic + sha256 split
- 数据集划分: 80/10/10 class-stratified
- 模型延迟加载: 首次 detect 才加载, 避免 GUI 启动延迟
- thread.quit() + wait() + deleteLater() 完整生命周期
- matplotlib.use("Agg") 在 import pyplot 之前 (避免 Qt 冲突)
- 所有颜色/字体/尺寸常量集中管理 (theme.py + app_config.py)

## 6. 可演示形态

- 单图检测 demo: 选图 → 点检测 → 21ms 后框画出 (crazing/inclusion 等)
- 1800 张批处理: 选文件夹 → 进度条实时更新 (每 10 张) → 33s 完成
- 取消机制实演: 批处理中点停止 → 18ms 响应 → 部分 CSV 写入
- HTML 报表导出: 批处理完 → 点导出 → 0.7s 生成 → 浏览器自动打开
- ONNX 推理对比: 同一张图 PyTorch GPU vs ONNX CPU 输出完全一致

## 7. git commit 历史（按时间倒序）

```
2ab4070 Day 15 P1+P2: HTML polish + ONNX export + benchmark (25 FPS CPU, consistency < 0.06px)
7221152 Day 14: HTML report with matplotlib charts, 1800 imgs in 0.7s, base64 self-contained
889c347 Day 13 closing: batch processing + per-class threshold + 11/11 validation, 1800 imgs 33.2s, cancel 18ms
8b6f441 Day 13: batch processing complete, 1800 imgs 33.7s, cancel 19ms, per-class threshold
a734601 Day 12: single-image detection + threaded worker, GPU 21.5ms, 5/5 acceptance
9455823 Day 11: PyQt6 SCADA skeleton complete, 7/7 acceptance passed
6a56380 Day 11 prep: env migration to pytorch env, base cleanup, Insight #5 logged
c96b9da Week 2 U-Net: refined v2 training + Insight #4 root cause analysis
9cff172 Day 5: test set终值 0.745, 分尺度+bad case分析, Insight #2/#3 root cause analysis
d966d61 Day 1-4: project init, EDA, split, training, 5 ablation experiments
```

## 8. DEVLOG 全部段落标题

```
# DEVLOG
## Day 1 — 2026-06-02：项目初始化
## Day 2 — 2026-06-03：数据探索（EDA）
## Day 2.5 — 2026-06-03：固定数据集 split
## Day 3 — 2026-06-03：YOLOv8 训练跑通
## Day 4 — 2026-06-03：调参实验
## Insight #1: 评测口径不统一 + mosaic 对纹理类有害
## Day 5 — 2026-06-03：Test Set 终值 + Bad Case
## Insight #2: val 0.800 vs test 0.745 的 -0.055 差距溯源
## Insight #3: 分尺度 mAP 评测的样本数警戒线
## Insight #4: U-Net 弱监督分割的两次踩坑 + 多类训练负迁移发现
## Insight #5: Python 环境管理踩坑 + 迁移到独立 env
## Day 11: PyQt6 SCADA 风主窗口骨架完工 — 2026-06-04
## Day 12: 单图检测 + 多线程 Worker 实战 — 2026-06-04
## Day 13: 批处理 Worker + per-class threshold + 全量验证 — 2026-06-04
## 日志格式模板
## Day 14: HTML 报表 + matplotlib 可视化 — 2026-06-04
## Day 15 (Part 1): HTML Polish + ONNX 导出 + 性能基准 — 2026-06-04
```

DEVLOG 总计: 16 个 ## 级标题 (含 12 Insight + 天级总结 + 模板)。
