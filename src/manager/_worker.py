"""推理 Worker。

属于 Manager 层——QObject 子类，运行在子线程。
不继承 QThread，通过 moveToThread 搬移到子线程执行。
"""

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from src.algo.postprocess import extract_detections
from src.algo.yolo_detector import YoloDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InferenceWorker(QObject):
    """单图推理 Worker。

    Manager 主线程创建本对象 → moveToThread(thread) → thread.started
    信号触发 run()。Worker 本身不创建也不拥有线程。

    信号：
        finished(dict): 推理成功，携带 extract_detections() 结果
        error(str): 推理异常，携带错误消息
    """

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancel_flag = False
        self._detector: YoloDetector | None = None

    def run(self, image: np.ndarray, conf: float, iou: float) -> None:
        """执行推理（由 thread.started 信号触发）。

        Args:
            image: BGR numpy array
            conf: 置信度阈值
            iou: NMS IoU 阈值
        """
        try:
            # 1. 检查取消标志
            if self._cancel_flag:
                logger.debug("Worker: cancelled before predict")
                result = extract_detections(
                    None, image.shape[:2], 0.0, cancelled=True
                )
                self.finished.emit(result)
                return

            # 2. 验证模型可用
            if self._detector is None:
                self.error.emit("Worker: detector not set")
                return

            # 3. 执行推理
            ultralytics_result, elapsed_ms = self._detector.predict(
                image, conf=conf, iou=iou
            )

            # 4. 检查取消标志（推理完成后、emit 前）
            if self._cancel_flag:
                logger.debug("Worker: cancelled after predict")
                result = extract_detections(
                    None, image.shape[:2], elapsed_ms, cancelled=True
                )
                self.finished.emit(result)
                return

            # 5. 提取结构化结果并发送
            result = extract_detections(
                ultralytics_result, image.shape[:2], elapsed_ms
            )
            logger.debug(
                f"Worker: finished — {result['num_detections']} detections "
                f"in {elapsed_ms:.1f}ms"
            )
            self.finished.emit(result)

        except Exception as exc:
            logger.error(f"Worker: exception — {exc}")
            self.error.emit(str(exc))

    def cancel(self) -> None:
        """设置取消标志。由 Manager.stop() 在主线程调用，线程安全。"""
        self._cancel_flag = True
        logger.debug("Worker: cancel flag set")
