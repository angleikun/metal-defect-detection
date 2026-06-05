"""批处理推理管理器。

属于 Manager 层——掌管批处理线程生命周期。
接收共享的 YoloDetector 引用，不创建新模型。
Day 13: 从空壳填实，5 步 connect 套路。
"""

from pathlib import Path

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import QMessageBox

from src.algo.yolo_detector import YoloDetector
from src.config.app_config import PER_CLASS_CONF, MAX_BATCH_FILES
from src.manager._batch_worker import BatchWorker
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BatchManager(QObject):
    """批量检测推理管理器。

    信号：
        batch_progress(int, int, str):  (completed, total, filename)
        batch_done(dict):               汇总 dict + csv_path
        batch_cancelled(int):           已处理张数
        batch_error(str):              异常消息
    """

    batch_progress = pyqtSignal(int, int, str)
    batch_done = pyqtSignal(dict)
    batch_cancelled = pyqtSignal(int, str)
    batch_error = pyqtSignal(str)

    def __init__(self, detector: YoloDetector, parent=None):
        """初始化。

        Args:
            detector: 已加载的 YoloDetector 实例（共享引用）
            parent: Qt parent
        """
        super().__init__(parent)
        self._detector = detector
        self._thread: QThread | None = None
        self._worker: BatchWorker | None = None
        self._busy = False
        logger.debug("BatchManager created (detector shared)")

    # ── 公开 API ────────────────────────────────────────────

    def is_busy(self) -> bool:
        """返回是否正在批处理中。"""
        return self._busy

    def run_batch(
        self,
        folder_path: str,
        conf: float,
        iou: float,
        per_class_conf: dict | None = None,
        extensions: tuple[str, ...] = (".jpg", ".jpeg"),
    ) -> None:
        """启动批量推理（异步）。

        Args:
            folder_path: 图像文件夹路径
            conf: YOLO 推理置信度阈值
            iou: NMS IoU 阈值
            per_class_conf: per-class 后处理阈值
            extensions: 允许的文件扩展名元组
        """
        if self._busy:
            logger.warning("BatchManager: already busy")
            return

        folder = Path(folder_path)
        if not folder.is_dir():
            self.batch_error.emit(f"不是有效文件夹: {folder_path}")
            return

        image_paths = sorted(
            str(p) for p in folder.rglob("*")
            if p.suffix.lower() in extensions
        )
        logger.info(
            f"BatchManager: scanned {folder_path} (recursive), "
            f"found {len(image_paths)} images"
        )

        # 空文件夹检查
        if len(image_paths) == 0:
            QMessageBox.information(
                None, "批处理",
                f"文件夹内未找到图像文件:\n{folder_path}\n\n"
                f"支持的格式: {', '.join(extensions)}"
            )
            self._busy = False
            return

        # 文件数上限检查
        if len(image_paths) > MAX_BATCH_FILES:
            reply = QMessageBox.warning(
                None, "批处理 — 文件数量警告",
                f"文件夹包含 {len(image_paths)} 张图像，"
                f"超过推荐上限 {MAX_BATCH_FILES}。\n\n"
                f"继续处理可能导致长时间运行。\n\n是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                self._busy = False
                return

        self._busy = True
        try:
            self._start_batch(image_paths, conf, iou, per_class_conf)
        except Exception as exc:
            self._busy = False
            self.batch_error.emit(f"启动批处理失败: {exc}")

    def stop(self) -> None:
        """取消当前批处理。"""
        if self._worker is not None:
            self._worker.cancel()
            logger.info("BatchManager: stop requested")

    # ── 线程管理 ────────────────────────────────────────────

    def _start_batch(
        self,
        image_paths: list[str],
        conf: float,
        iou: float,
        per_class_conf: dict | None,
    ) -> None:
        """创建线程 + BatchWorker + 5 步 connect + 启动。"""
        if per_class_conf is None:
            per_class_conf = PER_CLASS_CONF

        self._thread = QThread(self)
        self._worker = BatchWorker()
        self._worker._detector = self._detector
        self._worker.moveToThread(self._thread)

        # 5 步 connect
        # Step 1: thread 启动 → worker 执行
        self._thread.started.connect(
            lambda: self._worker.run(image_paths, conf, iou, per_class_conf)
        )
        # Step 2: progress → 透传到 UI
        self._worker.progress.connect(self.batch_progress)
        # Step 3: batch_finished → Manager 回调
        self._worker.batch_finished.connect(self._on_batch_finished)
        # Step 4: cancelled → Manager 回调
        self._worker.cancelled.connect(self._on_cancelled)
        # Step 5: error → Manager 回调
        self._worker.error.connect(self._on_error)
        # 无论成败/取消，停止线程
        self._worker.batch_finished.connect(self._thread.quit)
        self._worker.cancelled.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        # 线程停止 → 回收资源
        self._thread.finished.connect(self._cleanup)

        logger.info(f"BatchManager: starting {len(image_paths)} images")
        self._thread.start()

    # ── 回调（主线程执行） ─────────────────────────────────

    def _on_batch_finished(self, summary: dict) -> None:
        logger.info(
            f"BatchManager: done — {summary['defect_count']}/"
            f"{summary['total']} with defects, "
            f"avg {summary['avg_time_ms']}ms, "
            f"csv={Path(summary['csv_path']).name}"
        )
        self.batch_done.emit(summary)

    def _on_cancelled(self, count: int, csv_path: str) -> None:
        logger.info(
            f"BatchManager: cancelled after {count} images"
            + (f", csv={Path(csv_path).name}" if csv_path else "")
        )
        self.batch_cancelled.emit(count, csv_path)

    def _on_error(self, msg: str, tb: str) -> None:
        logger.error(f"BatchManager: error — {msg}")
        self.batch_error.emit(msg)

    def _cleanup(self) -> None:
        """线程回收。"""
        logger.debug("BatchManager: cleaning up thread")
        self._busy = False

        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None

        logger.debug("BatchManager: cleanup done")
