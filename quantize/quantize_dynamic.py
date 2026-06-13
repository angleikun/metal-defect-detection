"""
quantize_dynamic.py
ONNX 动态量化：FP32 -> INT8 (无符号)
"""

import os
import time
from onnxruntime.quantization import quantize_dynamic, QuantType

FP32_MODEL = "yolov8n_neu_fp32.onnx"
INT8_MODEL = "yolov8n_neu_int8.onnx"

if not os.path.exists(FP32_MODEL):
    raise FileNotFoundError(f"找不到 ONNX FP32 模型: {FP32_MODEL}")

print(f"[1/3] 输入: {FP32_MODEL}")
size_fp32 = os.path.getsize(FP32_MODEL) / 1024 / 1024
print(f"      FP32 大小: {size_fp32:.2f} MB")

print(f"\n[2/3] 执行动态量化 (INT8 无符号)")
print(f"      weight_type=QuantType.QUInt8")
print(f"      量化中... ", end="", flush=True)

t0 = time.time()
quantize_dynamic(
    model_input=FP32_MODEL,
    model_output=INT8_MODEL,
    weight_type=QuantType.QUInt8,
)
elapsed = time.time() - t0
print(f"完成 ({elapsed:.1f}s)")

print(f"\n[3/3] 验证产物")
size_int8 = os.path.getsize(INT8_MODEL) / 1024 / 1024
ratio = size_fp32 / size_int8
saved = size_fp32 - size_int8
saved_pct = (1 - size_int8 / size_fp32) * 100
print(f"      INT8 大小: {size_int8:.2f} MB")
print(f"      压缩比:   {ratio:.2f}x")
print(f"      节省:     {saved:.2f} MB ({saved_pct:.1f}%)")
print("\n量化完成")
