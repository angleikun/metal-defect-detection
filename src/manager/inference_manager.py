"""单图推理管理器。

属于 Manager 层——掌管模型生命周期 + 线程生命周期。
对外暴露 detect() 和 stop()，通过信号与 UI 层通信。
"""

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from src.algo.yolo_detector import YoloDetector
from src.config.app_config import YOLO_BEST
from src.manager._worker import InferenceWorker
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InferenceManager(QObject):
    """单图检测推理管理器。

    信号：
        inference_done(dict):   推理完成，携带 extract_detections() 结果
        error_occurred(str):    异常信息
        model_loaded(bool):     模型加载完成
    """

    inference_done = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    model_loaded = pyqtSignal(bool)

    def __init__(self, model_path: str | None = None, parent=None):
        """初始化管理器。

        Args:
            model_path: 模型文件路径，默认读取 app_config.YOLO_BEST
            parent: Qt parent
        """
        super().__init__(parent)
        self._model_path = str(model_path) if model_path else str(YOLO_BEST)
        self._detector: YoloDetector | None = None
        self._thread: QThread | None = None
        self._worker: InferenceWorker | None = None
        self._busy = False
        logger.debug("InferenceManager created (model not loaded)")

    # ── 公开 API ────────────────────────────────────────────

    def is_model_ready(self) -> bool:
        """返回模型是否已加载。"""
        return self._detector is not None

    def is_busy(self) -> bool:
        """返回是否正在推理中。"""
        return self._busy

    def detect(self, image: np.ndarray, conf: float, iou: float) -> None:
        """启动单图推理（异步）。

        Args:
            image: BGR numpy array
            conf: 置信度阈值
            iou: NMS IoU 阈值
        """
        if self._busy:
            logger.warning("InferenceManager: already busy, ignoring detect")
            return

        try:
            self._busy = True
            self._ensure_model_loaded()
            self._start_inference(image, conf, iou)
        except Exception as exc:
            self._busy = False
            logger.error(f"InferenceManager: failed to start — {exc}")
            self.error_occurred.emit(f"启动推理失败: {exc}")

    def stop(self) -> None:
        """取消当前推理（异步，不阻塞主线程）。"""
        if self._worker is not None:
            self._worker.cancel()
            logger.info("InferenceManager: stop requested")

    # ── 私有方法 ────────────────────────────────────────────

    def _ensure_model_loaded(self) -> None:
        """延迟加载模型（首次 detect 时，在主线程执行）。"""
        if self._detector is not None:
            return
        logger.info(f"Loading model: {self._model_path}")
        self._detector = YoloDetector(self._model_path)
        self.model_loaded.emit(True)
        logger.info(f"Model loaded on {self._detector.device}")

    def _start_inference(self, image: np.ndarray, conf: float, iou: float) -> None:
        """创建线程 + Worker + 5 步 connect + 启动。"""
        # 1. 创建线程
        self._thread = QThread(self)
        # 2. 创建 Worker
        self._worker = InferenceWorker()
        self._worker._detector = self._detector  # 传入模型引用
        # 3. 搬移到子线程
        self._worker.moveToThread(self._thread)

        # 4. 5 步 connect
        # Step 1: thread 启动 → worker 执行
        self._thread.started.connect(
            lambda: self._worker.run(image, conf, iou)
        )
        # Step 2: worker 完成 → Manager 回调
        self._worker.finished.connect(self._on_worker_finished)
        # Step 3: worker 异常 → Manager 回调
        self._worker.error.connect(self._on_worker_error)
        # Step 4: 无论成败，停止线程
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        # Step 5: 线程停止 → 回收资源
        self._thread.finished.connect(self._cleanup)

        # 5. 启动
        logger.info(
            f"InferenceManager: starting inference "
            f"(conf={conf:.2f}, iou={iou:.2f})"
        )
        self._thread.start()

    # ── 回调（主线程执行） ─────────────────────────────────

    def _on_worker_finished(self, result: dict) -> None:
        """Worker 推理完成回调。"""
        if result.get("cancelled", False):
            logger.info("InferenceManager: inference cancelled")
        else:
            logger.info(
                f"InferenceManager: done — {result['num_detections']} "
                f"detections in {result['inference_time_ms']}ms"
            )
        self.inference_done.emit(result)

    def _on_worker_error(self, msg: str) -> None:
        """Worker 异常回调。"""
        logger.error(f"InferenceManager: worker error — {msg}")
        self.error_occurred.emit(msg)

    def _cleanup(self) -> None:
        """线程回收（由 thread.finished 信号触发）。"""
        logger.debug("InferenceManager: cleaning up thread")
        self._busy = False

        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None

        logger.debug("InferenceManager: cleanup done")
