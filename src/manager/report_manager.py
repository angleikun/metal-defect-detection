"""报表导出管理器。

属于 Manager 层——协调检测结果导出为各种格式。
当前阶段为信号声明 + 方法空壳。
"""

from PyQt6.QtCore import QObject, pyqtSignal

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReportManager(QObject):
    """检测报表导出管理器。

    职责（后续实现）：
    - CSV 导出（pandas / csv）
    - Excel 导出（openpyxl）
    - PDF 报表（reportlab）

    信号：
        report_ready: 报表生成完成，携带文件路径
        report_error: 报表错误信息
    """

    report_ready = pyqtSignal(str)
    report_error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        logger.debug("ReportManager.__init__ (not implemented)")

    def export_csv(self, results: dict, output_path: str) -> None:
        """导出 CSV 报表（not implemented）。

        Args:
            results: 检测结果字典
            output_path: 输出 CSV 文件路径
        """
        logger.debug(f"ReportManager.export_csv('{output_path}') not implemented")

    def export_excel(self, results: dict, output_path: str) -> None:
        """导出 Excel 报表（not implemented）。

        Args:
            results: 检测结果字典
            output_path: 输出 XLSX 文件路径
        """
        logger.debug(f"ReportManager.export_excel('{output_path}') not implemented")

    def export_pdf(self, results: dict, output_path: str) -> None:
        """导出 PDF 报表（not implemented）。

        Args:
            results: 检测结果字典
            output_path: 输出 PDF 文件路径
        """
        logger.debug(f"ReportManager.export_pdf('{output_path}') not implemented")
