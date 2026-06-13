# YOLOv8n ONNX 量化实验

将主项目训练得到的 YOLOv8n NEU-DET 模型（6 类金属缺陷检测）从 PyTorch 部署形态压缩为 INT8 ONNX，并在嵌入式 Linux 部署预研场景下评估量化收益与代价。

## 实验结论速览

| | FP32 .pt | FP32 .onnx | INT8 .onnx |
|---|---|---|---|
| 文件大小 | 5.97 MB | (11.70 MB, 不入 git) | **3.20 MB** |
| mAP@0.5 | 0.770 | 0.751 | 0.646 |
| mAP@0.5:0.95 | 0.457 | 0.442 | 0.357 |
| 单图推理（yolo val, CPU）| 101.9 ms | **55.6 ms** | 66.5 ms |

**三个核心发现**：

1. **意外收益**：FP32 ONNX 相比 .pt 推理快 46%（ORT 算子融合带来的纯增益）
2. **量化反慢**：INT8 在本机 AMD Ryzen 7 6800H 上比 FP32 ONNX 慢 ~20%——根因是 Zen 3+ 缺 AVX-VNNI 指令支持（详见 `verification_report.md`）
3. **精度损失偏大**：INT8 mAP50 损失 0.105，超出 per-tensor 动态量化的典型范围；通过排除 Detect 层做证伪实验确认根因不在 Detect 层敏感性，真实瓶颈在别处（详见 `quantization_v1_vs_v2_comparison.md`）

## 目录结构

```
quantize/
├── README.md                                    本文档
├── verification_report.md                       量化生效验证（含硬件论据 + 局限性）
├── quantization_v1_vs_v2_comparison.md          v1 vs v2 对比实验（B1 假说证伪）
│
├── export_onnx.py                               PyTorch → ONNX FP32 导出
├── quantize_dynamic.py                          ONNX FP32 → INT8 动态量化 (v1)
├── quantize_dynamic_v2.py                       排除 Detect 层的 v2 量化（证伪实验）
├── benchmark_latency.py                         CPU 延迟 benchmark（50 iters，单线程）
│
├── yolov8n_neu_fp32.pt                          训练得到的 PyTorch 权重（5.97 MB）
├── yolov8n_neu_int8.onnx                        INT8 量化产物 v1（3.20 MB）
│
└── figures/                                     yolo val 输出的关键图
    ├── v1_fp32_pt_confusion_matrix.png
    ├── v2_fp32_onnx_PR_curve.png
    ├── v3_int8_v1_confusion_matrix.png
    └── v3_int8_v1_PR_curve.png
```

## 复现实验

环境要求：
- Python 3.10+
- PyTorch 2.5.1+
- Ultralytics 8.4.60+
- ONNX Runtime 1.23+

复现步骤：

```bash
# 1. 导出 FP32 ONNX
python export_onnx.py

# 2. 动态量化为 INT8
python quantize_dynamic.py

# 3. 延迟 benchmark
python benchmark_latency.py

# 4. mAP 评估（需主项目 configs/data.yaml）
yolo val task=detect model=yolov8n_neu_fp32.pt data=../configs/data.yaml imgsz=640 batch=1 device=cpu
yolo val task=detect model=yolov8n_neu_int8.onnx data=../configs/data.yaml imgsz=640 batch=1 device=cpu
```

## 硬件 / 软件环境

实验在以下环境完成：
- CPU：AMD Ryzen 7 6800H（Zen 3+，无 AVX-VNNI）
- OS：Windows 11
- Python 3.10.20
- onnxruntime 1.23.2

## 关键设计决策

**关于 simplify=True 的 trade-off**：onnx-simplifier 的算子融合（常量折叠 + Conv-BN-SiLU 融合）会引入 ~0.02 的 mAP50 数值差异。本实验保留 simplify=True，以换取 ORT 推理 46% 的速度增益——工业检测对 mAP ±0.02 的波动不敏感，但对延迟敏感，这是部署场景下的合理选择。

**关于 v2 实验的单变量隔离**：v2 量化脚本仅引入 `nodes_to_exclude` 一个改动，其余参数（weight_type、per_channel）与 v1 严格一致。这是为了在 v2 mAP 出现变化时能精确归因。结果显示 v2 与 v1 mAP50 差异仅 +0.001（噪声级），证伪"Detect 层敏感"假说。如果同时改动 per_channel 或其他参数，将无法做出此判断。

**关于纯 benchmark 与 yolo val 速度差异的说明**：`benchmark_latency.py` 报告的 INT8 推理 168 ms 与 `yolo val` 报告的 66.5 ms 不可直接对比——前者是纯推理 50 iters 平均，后者包含 ultralytics 框架的预处理/后处理。两者都展示"INT8 慢于 FP32"的结论，但绝对数字不能跨工具对比。

## 局限性

详见 `verification_report.md` 第 5 节。简要列出：
- 未做 ORT profiler 直接验证 kernel path 行为
- 未在支持 AVX-VNNI 的硬件上做对照实验
- 未测试 per_channel=True 和静态量化（QDQ）的效果

## 后续方向

- per_channel=True 量化对比
- 静态量化（带 calibration）
- 在 ARM 平台或 Intel 12 代+ 上的对照实验

## 关联文档

- 主项目 README：`../README.md`
- 主项目 DEVLOG：`../DEVLOG.md`（量化实验对应 Day 6）
