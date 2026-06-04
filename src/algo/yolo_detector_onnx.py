"""YOLO ONNX 推理器。

属于 Algorithm 层——纯算法，零 Qt 依赖。
使用 ultralytics ONNX 后端（内部通过 onnxruntime 执行推理），
与 YoloDetector 保持相同 detect() 接口。
Day 15 新增。
"""

import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.config.app_config import PER_CLASS_CONF
from src.algo.postprocess import extract_detections
from src.utils.logger import get_logger

logger = get_logger(__name__)


class YoloDetectorONNX:
    """YOLOv8 ONNX 推理器（CPU，onnxruntime）。

    与 YoloDetector.detect() 保持相同接口：
    - detect(image_path) → dict（boxes, classes, scores, ...）
    - 支持 per-class threshold 过滤

    ultralytics YOLO(onnx_path) 内部使用 onnxruntime CPUExecutionProvider。
    """

    def __init__(self, onnx_path: str | Path):
        """加载 ONNX 模型。

        Args:
            onnx_path: .onnx 模型文件路径

        Raises:
            FileNotFoundError: 模型文件不存在
        """
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        self._model_path = onnx_path
        self._model = YOLO(str(onnx_path))
        logger.info(f"YoloDetectorONNX loaded: {onnx_path.name} on CPU")

    def detect(
        self,
        image_path: str | Path,
        conf: float | None = None,
        iou: float = 0.45,
        per_class_conf: dict | None = None,
    ) -> dict:
        """便捷方法：读图 → ONNX 推理 → per-class 后处理 → 返回 dict。

        与 YoloDetector.detect() 签名完全一致。

        Args:
            image_path: 图像文件路径
            conf: 推理置信度阈值。None 时自动用 min(PER_CLASS_CONF.values())
            iou: NMS IoU 阈值
            per_class_conf: per-class 阈值过滤，None 时默认用 PER_CLASS_CONF

        Returns:
            extract_detections() 格式的 dict
        """
        if conf is None:
            conf = min(PER_CLASS_CONF.values())
        if per_class_conf is None:
            per_class_conf = PER_CLASS_CONF

        img = cv2.imread(str(image_path))
        if img is None:
            return _empty_result()

        # ONNX 推理（ultralytics 内部用 onnxruntime）
        t0 = time.perf_counter()
        results = self._model.predict(
            source=img,
            conf=conf,
            iou=iou,
            imgsz=640,
            verbose=False,
            device="cpu",
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return extract_detections(
            results[0], img.shape[:2], elapsed_ms,
            per_class_conf=per_class_conf,
        )


def _empty_result() -> dict:
    return {
        "boxes": [], "classes": [], "scores": [], "class_names": [],
        "inference_time_ms": 0.0, "num_detections": 0,
        "image_shape": (0, 0), "cancelled": False,
    }
