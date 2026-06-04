"""报表导出管理器。

属于 Manager 层——掌管报表生成线程生命周期。
Day 14 重写：从空壳填实，5 步 connect 套路。
"""

from pathlib import Path

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from src.config.app_config import REPORT_OUTPUT_DIR
from src.manager._report_worker import ReportWorker
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReportManager(QObject):
    """检测报表导出管理器。

    信号：
        report_ready(str):  生成的 HTML 文件路径
        report_error(str):  错误消息
    """

    report_ready = pyqtSignal(str)
    report_error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: ReportWorker | None = None
        logger.debug("ReportManager created")

    # ── 公开 API ────────────────────────────────────────────

    def generate_html_report(self, csv_path: str, output_dir: str | None = None) -> None:
        """启动报表生成（异步）。

        Args:
            csv_path: 批处理 CSV 文件路径
            output_dir: HTML 输出目录，默认 REPORT_OUTPUT_DIR
        """
        if self._thread is not None:
            logger.warning("ReportManager: already generating")
            return

        if output_dir is None:
            output_dir = str(REPORT_OUTPUT_DIR)

        # 验证 CSV 存在
        if not Path(csv_path).is_file():
            self.report_error.emit(f"CSV 文件不存在: {csv_path}")
            return

        try:
            self._start_report(csv_path, output_dir)
        except Exception as exc:
            self.report_error.emit(f"启动报表生成失败: {exc}")

    # ── 线程管理（5 步 connect） ────────────────────────────

    def _start_report(self, csv_path: str, output_dir: str) -> None:
        """创建 QThread + ReportWorker + 5 步 connect + 启动。"""
        self._thread = QThread(self)
        self._worker = ReportWorker()
        self._worker.moveToThread(self._thread)

        # Step 1: thread 启动 → worker 执行
        self._thread.started.connect(
            lambda: self._worker.run(csv_path, output_dir)
        )
        # Step 2: worker 完成 → Manager 回调
        self._worker.finished.connect(self._on_report_finished)
        # Step 3: worker 异常 → Manager 回调
        self._worker.error.connect(self._on_report_error)
        # Step 4: 无论成败，停止线程
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        # Step 5: 线程停止 → 回收资源
        self._thread.finished.connect(self._cleanup)

        logger.info(f"ReportManager: generating report from {Path(csv_path).name}")
        self._thread.start()

    # ── 回调（主线程执行） ─────────────────────────────────

    def _on_report_finished(self, html_path: str) -> None:
        logger.info(f"ReportManager: done — {html_path}")
        self.report_ready.emit(html_path)

    def _on_report_error(self, msg: str) -> None:
        logger.error(f"ReportManager: error — {msg}")
        self.report_error.emit(msg)

    def _cleanup(self) -> None:
        """线程回收。"""
        logger.debug("ReportManager: cleaning up thread")

        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None

        logger.debug("ReportManager: cleanup done")
