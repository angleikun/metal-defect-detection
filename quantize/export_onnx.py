"""
export_onnx.py
将 PyTorch best.pt 模型导出为 ONNX FP32 格式

用法：
    python export_onnx.py
"""

from ultralytics import YOLO
import os

# 输入：PyTorch 模型
PT_MODEL = "yolov8n_neu_fp32.pt"

# 检查模型存在
if not os.path.exists(PT_MODEL):
    raise FileNotFoundError(f"找不到模型: {PT_MODEL}")

print(f"[1/3] 加载 PyTorch 模型: {PT_MODEL}")
model = YOLO(PT_MODEL)
print(f"      类别数 nc = {model.model.nc}")
print(f"      类别名 = {model.names}")

print(f"\n[2/3] 导出为 ONNX FP32 (imgsz=640, opset=13)")
# imgsz=640    : 输入分辨率 640x640（YOLOv8 默认）
# opset=13     : ONNX 算子集版本（兼容性较好，量化器支持完整）
# simplify=True: 用 onnxsim 简化计算图，提升推理速度
# dynamic=False: 固定 batch=1（嵌入式部署常用）
onnx_path = model.export(
    format='onnx',
    imgsz=640,
    opset=13,
    simplify=True,
    dynamic=False,
)
print(f"      ONNX 已导出: {onnx_path}")


print(f"\n[3/3] 验证产物")
size_mb = os.path.getsize("yolov8n_neu_fp32.onnx") / 1024 / 1024
print(f"      yolov8n_neu_fp32.onnx  →  {size_mb:.2f} MB")
print("\n✓ 导出完成")