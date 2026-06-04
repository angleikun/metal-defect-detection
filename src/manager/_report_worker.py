"""报表生成 Worker。

属于 Manager 层——QObject 子类，运行在子线程。
读取 CSV → 生成图表 → 渲染 HTML → emit finished。
Day 14 新增。
"""

from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from src.algo.chart_generator import (
    generate_class_distribution_chart,
    generate_time_distribution_chart,
    generate_confidence_histogram,
)
from src.algo.report_renderer import render_html_report
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReportWorker(QObject):
    """报表生成 Worker。QObject 子类，moveToThread 搬移到子线程。

    信号：
        finished(str): HTML 文件路径
        error(str):    错误消息
    """

    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self, csv_path: str, output_dir: str) -> None:
        """入口方法，由 thread.started 信号触发。

        在子线程依次执行：读 CSV → 生成 3 张图表 → 渲染 HTML → emit finished。

        Args:
            csv_path: 批处理 CSV 文件路径
            output_dir: HTML 输出目录
        """
        try:
            logger.info(f"ReportWorker: generating report from {Path(csv_path).name}")

            # 生成图表（base64 字符串）
            pie_b64 = generate_class_distribution_chart(csv_path)
            time_b64 = generate_time_distribution_chart(csv_path)
            conf_b64 = generate_confidence_histogram(csv_path)

            # 渲染 HTML
            timestamp = Path(csv_path).stem.replace("batch_", "").replace("crazing_", "")
            output_path = str(Path(output_dir) / f"report_{timestamp}.html")
            html_path = render_html_report(
                csv_path, pie_b64, time_b64, conf_b64, output_path,
            )

            size_kb = Path(html_path).stat().st_size / 1024
            logger.info(f"ReportWorker: done — {html_path} ({size_kb:.0f} KB)")
            self.finished.emit(html_path)

        except Exception as exc:
            logger.error(f"ReportWorker: exception — {exc}")
            self.error.emit(str(exc))
