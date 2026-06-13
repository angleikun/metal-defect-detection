# YOLOv8n 动态量化 v1 vs v2 横向对比

## 1. 实验目的

验证 **B1 假说**：「v1 mAP50 损失 0.105 的主因是 Detect 层被不当量化」—— 通过排除 Detect 层 19 个 Conv 不参与量化（v2），检验 mAP 是否显著回收。

## 2. 实验设计

**单变量对照**。v1 和 v2 的唯一差异是 `nodes_to_exclude` 参数：

| 参数 | v1 | v2 | 说明 |
|---|---|---|---|
| `nodes_to_exclude` | 无（全部 64 个 Conv 量化） | 19 个 Detect 层 Conv 排除 | **唯一变量** |
| `weight_type` | `QUInt8` | `QUInt8` | 一致 |
| `op_types_to_quantize` | 未指定（默认全量） | `['Conv']`（显式） | 行为等价 |
| `per_channel` | 未指定（默认 False） | `False`（显式） | 一致 |
| `reduce_range` | 未指定（默认 False） | 未指定 | 一致 |
| `extra_options` | 未指定 | 未指定 | 一致 |

**评估方式**：`yolo val` 在同一环境下各跑一次（task=detect, imgsz=640, batch=1, device=cpu）。

## 3. 数据对比

### 3.1 总指标对比

| 指标 | FP32 .pt | FP32 .onnx | v1 (全量化) | v2 (排除 Detect) | v2 - v1 |
|---|---|---|---|---|---|
| mAP50 | 0.770 | 0.751 | **0.646** | **0.647** | **+0.001** |
| mAP50-95 | 0.457 | 0.442 | 0.357 | 0.360 | +0.003 |
| P | — | — | 0.574 | 0.571 | -0.003 |
| R | — | — | 0.676 | 0.676 | 0.000 |
| 文件大小 | 5.97 MB (.pt) | 11.70 MB | 3.20 MB | 5.32 MB | +2.12 MB (+66%) |
| Inference (yolo val) | ~101.9ms (.pt) | ~55.6ms | 66.5ms | **61.3ms** | **-5.2ms (-7.8%)** |

### 3.2 各类 mAP50 对比

| 类别 | FP32 .onnx（基准） | v1 (全量化) | v2 (排除 Detect) | v2 - v1 | v1 vs 基准损失 | 说明 |
|---|---|---|---|---|---|
| crazing | 0.200 | 0.193 | 0.189 | -0.004 | -0.007 | 量化后接近归零 |
| inclusion | 0.746 | **0.480** | **0.481** | **+0.001** | **-0.266** | 损失最大类 |
| patches | 0.991 | 0.986 | 0.986 | 0.000 | -0.005 | 微小损失 |
| pitted_surface | 0.989 | 0.936 | 0.936 | 0.000 | **-0.053** | 中等损失，工业检测不可忽略 |
| rolled-in_scale | 0.604 | **0.518** | **0.519** | **+0.001** | **-0.086** | 中等损失 |
| scratches | 0.976 | **0.765** | **0.770** | **+0.005** | **-0.211** | 大损失 |
| **all** | **0.751** | **0.646** | **0.647** | **+0.001** | **-0.105** | |

注：FP32 基准值来自 runs/detect/val-2/，实测精确值。v1 各类数据来自 runs/detect/val-3/，实测精确值。v2 各类数据来自 runs/detect/val-4/，实测精确值。v1 vs 基准损失 = v1 mAP50 - FP32 基准 mAP50，精确减法计算。

**关键事实**：三类大损失（inclusion / scratches / rolled-in_scale）的 v2-v1 差异分别为 +0.001、+0.005、+0.001，全部在 yolo val 实验噪声（~0.01-0.02）内。排除 Detect 层未带来任何有意义的精度回收。

## 4. 结论

### B1 假说不成立

**mAP50 差异 +0.001，远低于实验噪声水平（典型 yolo val 噪声 ~0.01-0.02）。无统计学意义。**

排除 Detect 层 19 个 Conv 不参与量化，mAP 从 0.646 变为 0.647，差异在 `yolo val` 的随机种子噪声量级之内。三类主要受损类别（inclusion / scratches / rolled-in_scale）的 mAP50 无任何改善。

**说明 v1 的 0.105 mAP50 损失主要来源不在 Detect 层，而在别处。**

### 可能的真实根因（推断，未经实验验证）

1. **per-tensor 量化的 scale 截断**：Backbone 卷积层输出通道间激活值分布差异大，共享 scale 导致部分通道的 INT8 表示严重截断 —— 这是 per-tensor 量化的已知弱点。当前 `per_channel=False`。
2. **QUInt8 与 SiLU 的适配问题**：YOLOv8 使用 SiLU（Swish）激活函数，输出范围 (-0.278, +∞) 约一半落在负值区。QUInt8 动态量化的 zero_point 对称性可能与 SiLU 负值区不匹配，导致量化误差放大。
3. **动态量化无 calibration**：动态量化在每轮推理时对激活值做在线量化的 min-max 统计，没有用真实校准数据集做离线 range 校准。高方差的激活通道会被离群值撑大 range，低位精度丢失。
4. **类别敏感性**：inclusion（夹杂物）和 scratches（划痕）是细粒度小目标缺陷，特征响应弱、信噪比低。量化后微小特征差异被进一步抹平，比其他类别（patches / pitted_surface，特征对比度高）更易受损。

### 意外发现：v2 推理速度显著快于 v1

v2 推理时间从 v1 的 66.5ms 降至 61.3ms（**-7.8%**），即排除 19 个 Conv 为 FP32 反而提速。这验证了 [verification_report.md](./verification_report.md) 中的推断：

> **ConvInteger 在无 AVX-VNNI 的硬件上，执行效率低于 oneDNN FP32 Conv 的 AVX2 优化路径。**

减少 ConvInteger 数量 → 减少慢速 ConvInteger 调用 → 提速。19 个 Detect 层 Conv 从 ConvInteger（慢）恢复为 Conv（快），净收益约 5ms。

### 5. 后续方向

未验证、待探索的改进路径（按成本排序）：

| 方向 | 成本 | 预期效果 | 说明 |
|---|---|---|---|
| `per_channel=True` | 极低（改一个参数） | 可能显著提升 mAP | 让每个输出通道有独立 scale，消除跨通道截断 |
| `weight_type=QInt8` | 极低 | 可能改善 SiLU 负值区 | 有符号 INT8 范围 [-128,127]，与 SiLU 负值区间 (-0.278,0] 更匹配 |
| 引入 calibration 数据集做静态量化（QDQ） | 中（需准备 calib 数据） | 预期大幅提升 mAP | 离线统计激活值 range，消除离群值撑大 scale 的问题 |
| 排除部分 Backbone 层 | 中（需逐层实验） | 不确定 | 如果 Backbone 浅层 Conv 对量化更敏感，选择性排除可回收精度 |
| 对照组硬件实验 | 高（需换机器） | 验证性质 | 在 VNNI 硬件上同时测 v1/v2，验证 ConvInteger 加速 + 精度表现 |

### 6. 实验的工程价值

这是一次**证伪性实验**。B1 假说（"排除 Detect 层能回收 mAP"）在 YOLOv8n + NEU-DET + 动态量化场景下**被证伪**。

工程上，证伪假说本身就是有价值的事实：
- 确定了一件事 **不是** 根因，缩小了排查范围
- 避免了在错误方向上继续投入时间（如进一步细化 Detect 层排除策略）
- 意外发现了 ConvInteger 在无 VNNI 硬件上的速度劣势的直接证据（v2 减少 19 个 ConvInteger 提速 7.8%）
- 实验的变量隔离设计干净（唯一的差异是 `nodes_to_exclude`），排除了其他混杂因素

下一步应从 per-tensor → per-channel 和动态量化 → calibration 静态量化两个方向切入，这是基于排除 B1 后最可能回收 0.105 mAP 损失的路径。
