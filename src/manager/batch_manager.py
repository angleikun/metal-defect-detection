"""批处理推理管理器。

属于 Manager 层——协调批量图像推理与 GUI 进度通信。
当前阶段为信号声明 + 方法空壳，禁止加载模型或启动线程。
"""

from PyQt6.QtCore import QObject, pyqtSignal

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BatchManager(QObject):
    """批量检测推理管理器。

    职责（后续实现）：
    - 遍历文件夹内所有图像
    - 逐张调用推理，汇总统计
    - 通过 batch_progress 信号更新进度条

    信号：
        batch_progress: 当前进度 (completed, total)
        batch_done: 批量完成，携带汇总统计字典
        batch_error: 批量异常信息
    """

    batch_progress = pyqtSignal(int, int)
    batch_done = pyqtSignal(dict)
    batch_error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        logger.debug("BatchManager.__init__ (not implemented)")

    def run_batch(self, folder_path: str, conf_threshold: float = 0.25,
                  iou_threshold: float = 0.45) -> None:
        """遍历文件夹执行批量推理（not implemented）。

        Args:
            folder_path: 图像目录路径
            conf_threshold: 置信度阈值
            iou_threshold: IoU 阈值
        """
        logger.debug(f"BatchManager.run_batch('{folder_path}') not implemented")

    def stop(self) -> None:
        """停止当前批处理（not implemented）。"""
        logger.debug("BatchManager.stop() not implemented")
