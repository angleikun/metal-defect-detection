"""一次性脚本：导出 YOLO best.pt → best.onnx。Day 15。"""
from pathlib import Path
from ultralytics import YOLO
from src.config.app_config import YOLO_BEST

pt_path = Path(YOLO_BEST)
pt_size_mb = pt_path.stat().st_size / (1024 * 1024)
print(f"Source: {pt_path} ({pt_size_mb:.1f} MB)")

print("Exporting ONNX (opset=12, simplify=True, imgsz=640)...")
model = YOLO(str(pt_path))
result_path = model.export(format="onnx", opset=12, simplify=True, imgsz=640)
print(f"ONNX: {result_path}")

# 验证文件存在 + 大小合理（> 1 MB）
onnx_path = Path(result_path)
assert onnx_path.exists(), "ONNX file not created!"
size_mb = onnx_path.stat().st_size / (1024 * 1024)
assert size_mb > 1.0, f"ONNX file too small: {size_mb:.2f} MB"
print(f"Size: {size_mb:.2f} MB")
print(f"Export successful: {onnx_path}")
