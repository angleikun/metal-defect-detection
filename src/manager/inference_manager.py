"""单图推理管理器。

属于 Manager 层——协调模型推理与 GUI 信号通信。
当前阶段为信号声明 + 方法空壳，禁止加载模型或启动线程。
"""

from PyQt6.QtCore import QObject, pyqtSignal

from src.utils.logger import get_logger

logger = get_logger(__name__)


class InferenceManager(QObject):
    """单张图像检测推理管理器。

    职责（后续实现）：
    - 加载 YOLOv8/U-Net 模型
    - 对单张图像执行推理
    - 通过信号将结果传递给 GUI

    信号：
        inference_done: 推理完成，携带检测结果字典
        progress_updated: 推理进度（0-100）
        error_occurred: 异常信息
    """

    inference_done = pyqtSignal(dict)
    progress_updated = pyqtSignal(int)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        logger.debug("InferenceManager.__init__ (not implemented)")

    def run(self, image_path: str, conf_threshold: float = 0.25,
            iou_threshold: float = 0.45) -> None:
        """执行单图推理（not implemented）。

        Args:
            image_path: 图像路径
            conf_threshold: 置信度阈值
            iou_threshold: IoU 阈值
        """
        logger.debug(f"InferenceManager.run('{image_path}') not implemented")

    def stop(self) -> None:
        """停止当前推理（not implemented）。"""
        logger.debug("InferenceManager.stop() not implemented")
