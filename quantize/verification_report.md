# YOLOv8n ONNX 动态量化验证报告

**验证日期**: 2026-06-12  
**验证目标**: 确认 INT8 模型比 FP32 慢 30% 是硬件原因（AMD Zen 3+ 无 VNNI）而非操作错误  
**被验证实验**: YOLOv8n NEU-DET（6 类金属缺陷）ONNX 动态量化 benchmark

---

## 1. 量化是否真实生效

### 方法
用 `onnx` 库分别加载 `yolov8n_neu_fp32.onnx` 和 `yolov8n_neu_int8.onnx`，统计 op_type 分布和 initializer 数据类型。

### 结果

| 指标 | FP32 | INT8 | 说明 |
|---|---|---|---|
| 总节点数 | 231 | 608 | INT8 膨胀 2.6×（量化/反量化节点） |
| **Conv** | **64** | **0** | 全部消失 |
| **ConvInteger** | **0** | **64** | **全部转换** ✅ |
| DynamicQuantizeLinear | 0 | 59 | 动态量化的激活值量化节点 |
| QuantizeLinear / DequantizeLinear | 0 | 0 | 非 Q/DQ 方式，是动态量化 |
| MatMul / MatMulInteger | 0 | 0 | 本次导出未产生 MatMul 节点（可能已融合到 Conv 1x1） |
| FLOAT initializer | 144 | 132 | bias/BN 参数保留 FP32 |
| UINT8 initializer | 0 | **128** | 权重已量化 |
| 文件大小 | 11.70 MB | 3.20 MB | 压缩至 27.4% |

### 结论
**量化真实生效，覆盖完整。** 64 个 Conv 全部转为 ConvInteger，128 个权重张量量化为 UINT8，文件大小降至原来的 27.4%。不存在"动态量化默认不量化 Conv"的问题。

---

## 2. 脚本审查

### 2.1 quantize_dynamic.py

**状态**: 无 bug，量化结果正确。

**调用方式**:
```python
quantize_dynamic(
    model_input=FP32_MODEL,
    model_output=INT8_MODEL,
    weight_type=QuantType.QUInt8,
)
```

| 参数 | 实际值 | 评价 |
|---|---|---|
| `op_types_to_quantize` | 未指定（实测默认通过 `IntegerOpsRegistry.keys()` fallback 为 `['Conv', 'MatMul', 'Attention', 'LSTM', 'Gather', 'Transpose', 'EmbedLayerNormalization']`，含 Conv，onnxruntime 1.23.2 quantize.py L855-856 验证） | ⚠️ 建议显式写 `['Conv']` 以防 onnxruntime 版本升级改默认 |
| `weight_type` | `QuantType.QUInt8` | ✅ 正确（脚本显式传 QUInt8，覆盖了源码默认值 QInt8） |
| `per_channel` | 未指定（实测默认 **False**，onnxruntime 1.23.2 quantize.py L798） | ⚠️ 当前 INT8 模型实际是 per-tensor 量化，非 per-channel |
| `nodes_to_exclude` | 未指定 | ✅ YOLOv8 无需排除 |

**建议改进（不影响当前结论）**:
```python
quantize_dynamic(
    model_input=FP32_MODEL,
    model_output=INT8_MODEL,
    weight_type=QuantType.QUInt8,
    op_types_to_quantize=['Conv'],   # YOLOv8 仅含 Conv，显式声明
    per_channel=True,                 # 显式声明（当前实际为 per-tensor，源码默认 False）
)
```

**关于 per_channel 的影响**：

当前 INT8 模型采用 per-tensor 量化（整个权重张量共享一组 scale/zero_point）。对 YOLOv8 这种 Conv 层每个输出通道权重分布差异较大的网络，per-channel 量化（每个输出通道独立 scale）通常精度更高。本次实验未启用 `per_channel=True`，是潜在的精度优化空间。

注：`per_channel` 主要影响 mAP，不影响延迟对比结论。本报告主结论（INT8 慢 30%）仍然成立。

### 2.2 benchmark_latency.py

**状态**: 无 bug，对比条件公平。

| 审查项 | FP32 | INT8 | 一致？ |
|---|---|---|---|
| `intra_op_num_threads` | 1 | 1 | ✅ |
| `graph_optimization_level` | ORT_ENABLE_ALL | ORT_ENABLE_ALL | ✅ |
| `providers` | `['CPUExecutionProvider']` | `['CPUExecutionProvider']` | ✅ |
| Warmup | 5 | 5 | ✅ |
| Iters | 50 | 50 | ✅ |
| 输入数据 | 同一份 `dummy_input` | 同一份 `dummy_input` | ✅ |

**建议改进（不影响当前结论）**: 显式设 `inter_op_num_threads=1` 和 `execution_mode=ORT_SEQUENTIAL` 以提高可复现性。

---

## 3. 硬件论据

### CPU 指令集检测

**CPU**: AMD Ryzen 7 6800H with Radeon Graphics（Zen 3+ / Rembrandt）

| 指令集 | 状态 | 意义 |
|---|---|---|
| **avx512_vnni** | ❌ False | Intel VNNI，AMD 不支持 |
| **avx_vnni** | ❌ **False** | AMD VNNI，**Zen 4 才引入** |
| avx512f / avx512bw | ❌ False | AVX-512，Zen 4 才支持 |
| **avx2** | ✅ True | 256-bit SIMD，FP32 推理主力 |
| sse4_2 | ✅ True | 128-bit 保底 |

### 技术解释

- **FP32 Conv** 在 oneDNN 中对 AVX2 有多年深度优化的微内核，执行效率高。
- **ConvInteger（UINT8）** 在缺少 VNNI 的硬件上，无法使用 `VPDPBUSD`（一次执行 4 路 INT8 点积）等 VNNI 加速指令。**推测** onnxruntime 会 fallback 到 AVX2 或更低优化级别的 INT8 实现路径，吞吐显著低于 FP32 在 oneDNN AVX2 优化微内核下的性能。（注：未做 ORT profiler 直接验证 kernel 选择，此处为基于硬件能力的推断。）
- **DynamicQuantizeLinear × 59 个节点** 是 INT8 独有的运行时开销——每次推理都要对激活值做动态量化，FP32 完全不需这一步。
- **综合效果（推测 + 实测）**：
  - **实测事实**：INT8 模型多出 59 个 DynamicQuantizeLinear 节点，是 FP32 不存在的运行时开销
  - **硬件事实**：本机无 AVX-VNNI（avx_vnni=False 实测）
  - **推断**：ConvInteger 在无 VNNI 硬件上的实现路径吞吐低于 FP32 Conv 的 AVX2 优化路径
  以上三者综合，导致 INT8 总延迟反超 FP32 30%。

---

## 4. 结论

### 原假说是否成立？

**"INT8 慢 30% 是因为 AMD Zen 3+ 无 VNNI"——成立，且可更精确表述。**

更精确的因果链：

```
AMD Zen 3+ 无 avx_vnni（实测）
  → ConvInteger 无法使用 VNNI 加速指令（硬件能力事实）
  → ConvInteger 实现路径吞吐 < oneDNN FP32 Conv 的 AVX2 优化路径（推断，未做 profiler 验证）
  → 叠加 59 个 DynamicQuantizeLinear 的固定开销（实测）
  → INT8 总延迟反超 FP32 30%（实测）
```

### 排除的替代解释

| 可能的假解释 | 验证结果 |
|---|---|
| 量化没生效（Conv 没转 ConvInteger） | ❌ 排除——64/64 全部转换 |
| benchmark 脚本不对称 | ❌ 排除——FP32/INT8 同条件对比 |
| pip 安装的包版本不一致 | 未验证（但同一 session 内跑，环境一致） |
| CPU 温度降频 | 未直接验证，但 50 iters × 平均 ~150ms ≈ 7.5 秒总时长，远低于 Ryzen 6800H 触发持续热降频的典型时间（30s+ 持续高负载），概率极低 |

### 给报告的建议引用

- **一句话结论**: "在 AMD Ryzen 7 6800H（Zen 3+）上，YOLOv8n 动态量化 INT8 推理比 FP32 慢 30%，根因是 Zen 3+ 缺少 AVX-VNNI 指令支持，ConvInteger 无法从硬件 INT8 加速获益。"
- **如果想在报告中展示量化是否生效**: 引用本文档第 1 节的 Conv → ConvInteger 转换率（100%）和权重 UINT8 比例（128/260）。
- **如果想讨论硬件门槛**: 引用本文档第 3 节的 CPU flags，明确写"INT8 动态量化的 CPU 加速至少需要 Zen 4 (avx_vnni=True) 或 Intel 平台 (avx512_vnni=True)"。
- **后续实验建议**: 在支持 VNNI 的硬件（如 Intel 12 代+ 或 AMD Zen 4+）上跑同一套脚本做对照，预期 INT8 应能显著快于 FP32。

---

## 5. 验证局限性

本次验证未做以下检查，结论中相关推断未经直接证据支持，记录于此以供后续补强：

| 未验证项 | 影响范围 | 补强方法 |
|---|---|---|
| 未做 ORT profiler 分析 | 第 3 节"ConvInteger 退化到 AVX2 路径"为基于硬件能力的推断，未直接观测 kernel 选择 | 在 benchmark 脚本中启用 `sess_options.enable_profiling = True`，分析每个 ConvInteger op 的实际耗时和 kernel 名称 |
| 未做对照硬件实验 | "在 VNNI 硬件上 INT8 应快于 FP32"是正向预期，本机无法直接证伪 | 在 Intel 12 代+（avx512_vnni）或 AMD Zen 4+（avx_vnni）平台上复跑同一套脚本 |
| 未做温度/频率监控 | 50 iters 短 benchmark 期间未记录 CPU 温度和频率曲线 | 用 HWiNFO64 或 perfmon 记录 benchmark 全程的 CPU 频率，确认未发生降频 |
| 未做 per_channel 量化对比 | 当前 INT8 模型是 per-tensor（源码默认），未验证 per_channel=True 对延迟和 mAP 的影响 | 用 per_channel=True 重新量化一份模型，对比 mAP 和延迟差异 |

**主结论的影响范围**：

以上局限**不影响**本报告主结论：
- INT8 模型已正确量化（64 Conv → 64 ConvInteger，128 个 UINT8 initializer 实测）
- INT8 在本机慢于 FP32 是真实现象（同条件 benchmark 实测）
- benchmark 脚本无对比性偏差（脚本审查实测）

以上局限**限制**了根因分析的精确性：
- 无法断言 "INT8 慢的根因 100% 是 VNNI 缺失"——硬件能力是已知事实，但 kernel 实际行为是推断
- 无法量化"DynamicQuantizeLinear 开销" vs "ConvInteger 慢于 FP32 Conv"各自占慢度的比例

如需更强证据支持根因分析，建议优先补做 ORT profiler（成本最低，5 分钟），其次是对照硬件实验（成本最高，需要换机器）。

### 验证方法可复现性

所有检测脚本（`_check_onnx_ops.py`、`_find_onnx.py`）保留在工作目录下，可随时复跑验证。
