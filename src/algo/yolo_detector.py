"""YOLO 检测器封装。

属于 Algorithm 层——纯算法，零 Qt 依赖。
对 ultralytics YOLO 做薄封装，供 InferenceWorker 在子线程调用。

Day 12 修复：显式指定 device，避免 YOLO 默认加载到 CPU。
Day 13: 新增 detect() 便捷方法（predict + per-class 后处理）。
"""

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.config.app_config import MAX_DET, PER_CLASS_CONF
from src.algo.postprocess import extract_detections
from src.utils.logger import get_logger

logger = get_logger(__name__)


class YoloDetector:
    """YOLOv8 目标检测器。

    模型在 Manager 主线程加载，通过引用传入 Worker。
    Worker 子线程调用 predict() 执行推理。

    Day 12 修复：显式指定 device，避免 YOLO 默认加载到 CPU。
    """

    def __init__(self, model_path: str | Path, device: str = "auto"):
        """加载 YOLO 模型并移动到指定设备。

        Args:
            model_path: .pt 模型文件路径
            device: 'auto' / 'cuda' / 'cpu' / '0' 等。默认 'auto' 自动选 CUDA

        Raises:
            FileNotFoundError: 模型文件不存在
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # 自动选 device
        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        self._model_path = model_path
        self._model = YOLO(str(model_path))
        self._model.to(self._device)

        # 验证模型确实在目标设备上
        actual_device = str(next(self._model.model.parameters()).device)
        logger.info(
            f"YoloDetector loaded: {model_path.name} on {actual_device}"
        )

    def predict(
        self,
        image: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = MAX_DET,
        imgsz: int = 640,
    ) -> tuple:
        """对单张图像执行推理。

        Args:
            image: BGR 格式 numpy array（OpenCV 读入）
            conf: 置信度阈值
            iou: NMS IoU 阈值
            max_det: 最大检测数
            imgsz: 推理输入尺寸

        Returns:
            (ultralytics.Results, inference_time_ms)
            Results.xyxy[0] 为 numpy array [N, 6]: (x1, y1, x2, y2, conf, cls)
        """
        t0 = time.perf_counter()
        results = self._model.predict(
            source=image,
            conf=conf,
            iou=iou,
            max_det=max_det,
            imgsz=imgsz,
            verbose=False,
            device=self._device,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return results[0], elapsed_ms

    def detect(
        self,
        image_path: str | Path,
        conf: float | None = None,
        iou: float = 0.45,
        per_class_conf: dict | None = None,
    ) -> dict:
        """便捷方法：读图 → 推理 → per-class 后处理 → 返回干净 dict。

        Day 13 新增。供批处理 Worker 和快速测试使用。

        Args:
            image_path: 图像文件路径
            conf: 推理置信度阈值。None 时自动用 min(PER_CLASS_CONF.values())
            iou: NMS IoU 阈值
            per_class_conf: per-class 阈值过滤，None 时默认用 PER_CLASS_CONF

        Returns:
            extract_detections() 格式的 dict
        """
        # 推理用全局最低 conf 拿所有候选
        if conf is None:
            conf = min(PER_CLASS_CONF.values())
        if per_class_conf is None:
            per_class_conf = PER_CLASS_CONF

        img = cv2.imread(str(image_path))
        if img is None:
            return {
                "boxes": [], "classes": [], "scores": [], "class_names": [],
                "inference_time_ms": 0.0, "num_detections": 0,
                "image_shape": (0, 0), "cancelled": False,
            }

        raw_result, elapsed_ms = self.predict(img, conf=conf, iou=iou)
        return extract_detections(
            raw_result, img.shape[:2], elapsed_ms,
            per_class_conf=per_class_conf,
        )

    @property
    def model(self):
        """暴露底层 ultralytics YOLO 模型引用（只读）。"""
        return self._model

    @property
    def device(self) -> str:
        """返回模型所在设备。"""
        return self._device
