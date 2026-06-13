"""
quantize_dynamic_v2.py
ONNX 动态量化 v2：排除 Detect 层，保留 FP32 精度

与 v1 的关键改动：
  - 新增 nodes_to_exclude 参数，排除 YOLOv8 Detect 层 19 个 Conv 不量化
  - 检测头（cv2 bbox回归 + cv3 分类 + dfl）对动态量化敏感，
    v1 量化了全部 64 个 Conv 导致 mAP50 损失 0.105
  - 预期：v2 mAP 损失应小于 v1，体积略大于 v1

输出：yolov8n_neu_int8_v2.onnx（不覆盖 v1）
"""

import os
import time
from onnxruntime.quantization import quantize_dynamic, QuantType

# ============================================================
# Detect 层 19 个 Conv（不参与量化）
# YOLOv8n model.22: 3 检测尺度 × 2 分支(cv2/cv3) × 3 层 + 1 dfl
# ============================================================
DETECT_LAYER_CONVS = [
    # --- cv2 分支（bbox 回归）---
    "/model.22/cv2.0/cv2.0.0/conv/Conv",   # P3 scale, bbox layer 1
    "/model.22/cv2.0/cv2.0.1/conv/Conv",   # P3 scale, bbox layer 2
    "/model.22/cv2.0/cv2.0.2/Conv",        # P3 scale, bbox layer 3 (output)
    "/model.22/cv2.1/cv2.1.0/conv/Conv",   # P4 scale, bbox layer 1
    "/model.22/cv2.1/cv2.1.1/conv/Conv",   # P4 scale, bbox layer 2
    "/model.22/cv2.1/cv2.1.2/Conv",        # P4 scale, bbox layer 3 (output)
    "/model.22/cv2.2/cv2.2.0/conv/Conv",   # P5 scale, bbox layer 1
    "/model.22/cv2.2/cv2.2.1/conv/Conv",   # P5 scale, bbox layer 2
    "/model.22/cv2.2/cv2.2.2/Conv",        # P5 scale, bbox layer 3 (output)
    # --- cv3 分支（分类）---
    "/model.22/cv3.0/cv3.0.0/conv/Conv",   # P3 scale, cls layer 1
    "/model.22/cv3.0/cv3.0.1/conv/Conv",   # P3 scale, cls layer 2
    "/model.22/cv3.0/cv3.0.2/Conv",        # P3 scale, cls layer 3 (output)
    "/model.22/cv3.1/cv3.1.0/conv/Conv",   # P4 scale, cls layer 1
    "/model.22/cv3.1/cv3.1.1/conv/Conv",   # P4 scale, cls layer 2
    "/model.22/cv3.1/cv3.1.2/Conv",        # P4 scale, cls layer 3 (output)
    "/model.22/cv3.2/cv3.2.0/conv/Conv",   # P5 scale, cls layer 1
    "/model.22/cv3.2/cv3.2.1/conv/Conv",   # P5 scale, cls layer 2
    "/model.22/cv3.2/cv3.2.2/Conv",        # P5 scale, cls layer 3 (output)
    # --- dfl 分支（分布焦点损失）---
    "/model.22/dfl/conv/Conv",             # DFL convolutional layer
]

FP32_MODEL = "yolov8n_neu_fp32.onnx"
INT8_MODEL = "yolov8n_neu_int8_v2.onnx"  # v2 产物，不覆盖 v1

# ============================================================
# 第一阶段：执行动态量化
# ============================================================

if not os.path.exists(FP32_MODEL):
    raise FileNotFoundError(f"找不到 ONNX FP32 模型: {FP32_MODEL}")

print(f"[1/4] 输入: {FP32_MODEL}")
size_fp32 = os.path.getsize(FP32_MODEL) / 1024 / 1024
print(f"      FP32 大小: {size_fp32:.2f} MB")

print(f"\n[2/4] 执行动态量化 (INT8 无符号, v2: 排除 Detect 层)")
print(f"      weight_type=QuantType.QUInt8")
print(f"      op_types_to_quantize=['Conv']")
print(f"      per_channel=False (显式，per-tensor，与 v1 一致)")
print(f"      nodes_to_exclude={len(DETECT_LAYER_CONVS)} 个 Detect 层 Conv")
print(f"      量化中... ", end="", flush=True)

t0 = time.time()
quantize_dynamic(
    model_input=FP32_MODEL,
    model_output=INT8_MODEL,
    weight_type=QuantType.QUInt8,
    op_types_to_quantize=['Conv'],
    per_channel=False,
    nodes_to_exclude=DETECT_LAYER_CONVS,
)
elapsed = time.time() - t0
print(f"完成 ({elapsed:.1f}s)")

# ============================================================
# 第二阶段：验证产物
# ============================================================

print(f"\n[3/4] 验证产物")

import onnx
from collections import Counter

m = onnx.load(INT8_MODEL, load_external_data=False)
ops = Counter(n.op_type for n in m.graph.node)

size_int8 = os.path.getsize(INT8_MODEL) / 1024 / 1024
ratio = size_fp32 / size_int8
saved = size_fp32 - size_int8
saved_pct = (1 - size_int8 / size_fp32) * 100

print(f"      INT8 大小: {size_int8:.2f} MB")
print(f"      压缩比:   {ratio:.2f}x")
print(f"      节省:     {saved:.2f} MB ({saved_pct:.1f}%)")
print(f"      节点数:   Conv={ops.get('Conv', 0)}  ConvInteger={ops.get('ConvInteger', 0)}")

# ============================================================
# 第三阶段：验证 Detect 层排除是否生效
# ============================================================

print(f"\n[4/4] 验证 Detect 层排除是否正确")

# 检查 Detect 层每个节点在 v2 中的 op_type
detect_nodes_in_v2 = {
    n.name: n.op_type for n in m.graph.node if n.name in DETECT_LAYER_CONVS
}

# 必须全部为 Conv（不是 ConvInteger）
fail = []
for name in DETECT_LAYER_CONVS:
    actual = detect_nodes_in_v2.get(name, "NOT_FOUND")
    if actual != "Conv":
        fail.append((name, actual))

if fail:
    print(f"      ❌ 有 {len(fail)} 个节点未正确排除:")
    for name, op in fail:
        print(f"         {name}  →  {op}")
else:
    print(f"      ✅ 全部 {len(DETECT_LAYER_CONVS)} 个 Detect 层 Conv 保留为 Conv（未被量化）")

# 汇总
expected_convint = 64 - 19  # v1 全量 64 Conv, 排除 19 个
actual_convint = ops.get("ConvInteger", 0)
print(f"\n      预期:  ConvInteger={expected_convint},  Conv 保留={len(DETECT_LAYER_CONVS)}")
print(f"      实际:  ConvInteger={actual_convint},  Conv={ops.get('Conv', 0)}")
if actual_convint == expected_convint:
    print(f"      ✅ ConvInteger 数量正确 ({actual_convint})")
else:
    print(f"      ⚠️ ConvInteger 数量与预期不符（预期 {expected_convint}，实际 {actual_convint}）")

print("\n量化完成 (v2)")
