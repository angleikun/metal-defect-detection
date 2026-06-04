"""主窗口骨架 —— SCADA 风格三栏布局。UI 层，只组合不写算法逻辑。"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QSplitter,
    QListWidget, QStatusBar, QLabel,
)

from src.config.app_config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, IMAGE_DIR,
)
from src.ui.theme import (
    BACKGROUND, SURFACE, BORDER, TEXT_PRIMARY, TEXT_SECONDARY,
    GREEN_OK, FONT_FAMILY, FONT_SIZE_STATUS,
    LEFT_PANEL_WIDTH, RIGHT_PANEL_WIDTH, SPACING_MD, STATUS_BAR_HEIGHT,
)
from src.ui.image_viewer import ImageViewer
from src.ui.control_panel import ControlPanel
from src.ui.log_panel import LogPanel
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── 模块级样式常量 ───────────────────────────────────────────
_MENU_STYLE = f"""
    QMenuBar {{ background-color:{SURFACE}; color:{TEXT_PRIMARY};
        border-bottom:1px solid {BORDER}; font-family:{FONT_FAMILY};
        font-size:{FONT_SIZE_STATUS}px; }}
    QMenuBar::item:selected {{ background-color:{BACKGROUND}; }}
    QMenu {{ background-color:{SURFACE}; color:{TEXT_PRIMARY};
        border:1px solid {BORDER}; font-family:{FONT_FAMILY}; }}
    QMenu::item:selected {{ background-color:{BACKGROUND}; }}
"""

_LIST_STYLE = f"""
    QListWidget {{ background-color:{SURFACE}; color:{TEXT_PRIMARY};
        border:1px solid {BORDER}; font-family:{FONT_FAMILY};
        font-size:{FONT_SIZE_STATUS}px; }}
    QListWidget::item {{ padding:4px 8px; border-bottom:1px solid {BORDER}; }}
    QListWidget::item:selected {{ background-color:{BACKGROUND}; color:{GREEN_OK}; }}
    QListWidget::item:hover {{ background-color:{BACKGROUND}; }}
"""


class MainWindow(QMainWindow):
    """SCADA 主窗口：菜单栏 + 三栏(列表|图像|控制) + 日志面板 + 状态栏。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet(f"QMainWindow {{ background-color: {BACKGROUND}; }}")
        self._log_panel = None  # 在 _build_central 中设置

        self._build_menu_bar()
        self._build_central()
        self._build_status_bar()
        logger.info(f"MainWindow initialized ({WINDOW_WIDTH}×{WINDOW_HEIGHT})")

    # ── 菜单栏 ──────────────────────────────────────────────

    def _build_menu_bar(self) -> None:
        mb = self.menuBar()
        mb.setStyleSheet(_MENU_STYLE)
        log = lambda label: lambda: logger.info(f"Menu: {label}")

        file_menu = mb.addMenu("文件")
        file_menu.addAction(self._act("打开图像...", "Ctrl+O", log("打开图像...")))
        file_menu.addAction(self._act("打开文件夹...", "Ctrl+Shift+O", log("打开文件夹...")))
        file_menu.addSeparator()
        file_menu.addAction(self._act("导出报表...", "Ctrl+E", log("导出报表...")))
        file_menu.addSeparator()
        file_menu.addAction(self._act("退出", "Alt+F4", self.close))

        cfg_menu = mb.addMenu("配置")
        cfg_menu.addAction(self._act("模型设置...", None, log("模型设置...")))
        cfg_menu.addAction(self._act("报警阈值...", None, log("报警阈值...")))

        view_menu = mb.addMenu("视图")
        view_menu.addAction(self._act("重置布局", None, log("重置布局")))
        view_menu.addAction(self._act("清空日志", None, self._on_clear_log))

        alarm_menu = mb.addMenu("报警")
        alarm_menu.addAction(self._act("报警历史", None, log("报警历史")))
        alarm_menu.addAction(self._act("静音报警", None, log("静音报警")))

        help_menu = mb.addMenu("帮助")
        help_menu.addAction(self._act("关于", None, log("关于")))
        help_menu.addAction(self._act("使用手册", "F1", log("使用手册")))

    def _on_clear_log(self) -> None:
        if self._log_panel:
            self._log_panel.clear()

    @staticmethod
    def _act(text: str, shortcut: str | None, slot) -> QAction:
        action = QAction(text, None)  # parent=None, owned by menu
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        return action

    # ── 中央区域 ────────────────────────────────────────────

    def _build_central(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 水平三栏
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet(f"QSplitter::handle {{ background-color: {BORDER}; }}")

        # 左：图像列表
        self._image_list = QListWidget()
        self._image_list.setMaximumWidth(LEFT_PANEL_WIDTH)
        self._image_list.setStyleSheet(_LIST_STYLE)
        self._image_list.itemClicked.connect(
            lambda item: logger.info(f"List: clicked '{item.text()}'")
        )
        self._populate_image_list()
        splitter.addWidget(self._image_list)

        # 中：图像预览
        self._image_viewer = ImageViewer()
        splitter.addWidget(self._image_viewer)

        # 右：控制面板
        self._control_panel = ControlPanel()
        self._control_panel.setMaximumWidth(RIGHT_PANEL_WIDTH)
        splitter.addWidget(self._control_panel)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        splitter.setStretchFactor(2, 3)
        root.addWidget(splitter)

        # 底：日志面板
        self._log_panel = LogPanel()
        root.addWidget(self._log_panel)

    # ── 状态栏 ──────────────────────────────────────────────

    def _build_status_bar(self) -> None:
        status = QStatusBar()
        status.setStyleSheet(f"""
            QStatusBar {{ background-color:{SURFACE}; color:{TEXT_SECONDARY};
                border-top:1px solid {BORDER};
                font-family:{FONT_FAMILY}; font-size:{FONT_SIZE_STATUS}px; }}
        """)
        status.setFixedHeight(STATUS_BAR_HEIGHT)

        status_label = QLabel("● 就绪")
        status_label.setStyleSheet(f"color: {GREEN_OK}; padding-left: {SPACING_MD}px;")
        status.addWidget(status_label)
        status.addPermanentWidget(QLabel("Metal Defect Detection v1.0"))
        self.setStatusBar(status)

    # ── 图像列表 ────────────────────────────────────────────

    def _populate_image_list(self) -> None:
        """用 train 目录下前 30 张图像名填充（占位，不加载算法）。"""
        images = sorted(IMAGE_DIR.rglob("*.jpg"))
        for p in images[:30]:
            self._image_list.addItem(p.name)
