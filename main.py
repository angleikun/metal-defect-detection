"""应用入口。

启动 PyQt6 QApplication 并显示 SCADA 风主窗口。
所有项目命令必须先 `mamba activate pytorch`。
"""

import sys

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from src.ui.main_window import MainWindow
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    logger.info("Starting Metal Defect Detection SCADA...")

    app = QApplication(sys.argv)
    app.setApplicationName("MetalDefectDetection")

    # 全局高 DPI 支持
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    logger.info("MainWindow shown — entering event loop")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
