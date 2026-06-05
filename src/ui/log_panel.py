"""事件日志面板。

属于 UI 层——底部只读日志区域，接收 Logger 信号实时追加。
"""

from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import QPlainTextEdit, QSizePolicy

from src.ui.theme import (
    BACKGROUND,
    SURFACE,
    BORDER,
    TEXT_PRIMARY,
    CYAN_INFO,
    FONT_FAMILY,
    FONT_SIZE_LOG,
    SPACING_XS,
)
from src.utils.logger import log_signal, get_logger

logger = get_logger(__name__)


class LogPanel(QPlainTextEdit):
    """SCADA 风格事件日志面板。

    连接 Logger 模块的 log_signal，实现实时追加。
    只读模式，最大行数限制防止内存溢出。
    """

    MAX_LINES = 2000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(self.MAX_LINES)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.setFixedHeight(140)

        self._setup_font()
        self._setup_style()

        # 连接 Logger 信号
        log_signal.connect(self.append_text)

        # 初始消息
        logger.info("LogPanel initialized — logs stream active")

    def append_text(self, text: str) -> None:
        """追加一行日志文字。

        Args:
            text: 日志行（已含时间戳和级别前缀）
        """
        self.appendPlainText(text)
        # 自动滚动到底部
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.setTextCursor(cursor)

    def _setup_font(self) -> None:
        font = QFont(FONT_FAMILY, FONT_SIZE_LOG)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)

    def _setup_style(self) -> None:
        self.setStyleSheet(f"""
            QPlainTextEdit {{
                background-color: {SURFACE};
                color: {TEXT_PRIMARY};
                border: 1px solid {BORDER};
                border-radius: 2px;
                padding: {SPACING_XS}px;
                selection-background-color: {CYAN_INFO};
                selection-color: {BACKGROUND};
            }}
        """)
