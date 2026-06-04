"""主窗口骨架 —— SCADA 风格三栏布局。UI 层，只组合不写算法逻辑。
Day 12 增强：连接 InferenceManager ↔ UI 控件，端到端检测流程。"""

import cv2
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QSplitter,
    QListWidget, QListWidgetItem, QStatusBar, QLabel,
)

from src.config.app_config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, IMAGE_DIR, YOLO_BEST,
)
from src.manager.inference_manager import InferenceManager
from src.ui.theme import (
    BACKGROUND, SURFACE, BORDER, TEXT_PRIMARY, TEXT_SECONDARY,
    GREEN_OK, RED_ALARM, YELLOW_WARNING, CYAN_INFO,
    FONT_FAMILY, FONT_SIZE_STATUS,
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
        self._log_panel = None
        self._status_label = None
        self._current_bgr: np.ndarray | None = None

        self._build_menu_bar()
        self._build_central()
        self._build_status_bar()
        self._init_manager()
        self._connect_signals()
        logger.info(f"MainWindow initialized ({WINDOW_WIDTH}×{WINDOW_HEIGHT})")

    # ── Manager 初始化 ──────────────────────────────────────

    def _init_manager(self) -> None:
        """创建 InferenceManager（模型延迟加载，首次 detect 时加载）。"""
        self._inference_manager = InferenceManager(str(YOLO_BEST), self)
        # 持久化连接 Manager → UI
        self._inference_manager.inference_done.connect(self._on_result)
        self._inference_manager.error_occurred.connect(self._on_error)
        self._inference_manager.model_loaded.connect(self._on_model_loaded)

    # ── 信号接线 ────────────────────────────────────────────

    def _connect_signals(self) -> None:
        """UI 控件信号 ↔ 处理逻辑。"""
        self._image_list.itemClicked.connect(self._on_image_selected)
        self._control_panel.detect_clicked.connect(self._on_detect)
        self._control_panel.stop_clicked.connect(self._on_stop)
        self._control_panel.clear_clicked.connect(self._on_clear_boxes)

    def _set_status(self, text: str, color: str) -> None:
        """更新状态栏文字和颜色。"""
        self._status_label.setText(text)
        self._status_label.setStyleSheet(
            f"color: {color}; padding-left: {SPACING_MD}px;"
        )

    def _on_model_loaded(self, ok: bool) -> None:
        if ok:
            self._set_status("● 模型就绪", CYAN_INFO)

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
        action = QAction(text, None)
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

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet(f"QSplitter::handle {{ background-color: {BORDER}; }}")

        # 左：图像列表
        self._image_list = QListWidget()
        self._image_list.setMaximumWidth(LEFT_PANEL_WIDTH)
        self._image_list.setStyleSheet(_LIST_STYLE)
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

        self._status_label = QLabel()
        status.addWidget(self._status_label)
        self._set_status("● 就绪", GREEN_OK)
        status.addPermanentWidget(QLabel("Metal Defect Detection v1.0"))
        self.setStatusBar(status)

    # ── 图像列表 ────────────────────────────────────────────

    def _populate_image_list(self) -> None:
        """用 train 目录下前 30 张图像名填充（含完整路径）。"""
        images = sorted(IMAGE_DIR.rglob("*.jpg"))
        for p in images[:30]:
            item = QListWidgetItem(p.name)
            item.setData(Qt.ItemDataRole.UserRole, str(p))
            self._image_list.addItem(item)

    # ── UI 事件处理 ────────────────────────────────────────

    def _on_image_selected(self, item: QListWidgetItem) -> None:
        """列表点击 → 加载图像到 viewer + 清除旧检测框。"""
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path:
            return

        # 用 OpenCV 读取（BGR，与 YOLO 训练一致）
        self._current_bgr = cv2.imread(path)
        if self._current_bgr is None:
            logger.warning(f"Cannot read image: {path}")
            return

        self._image_viewer.load_image(path)
        self._control_panel.reset_stats()
        self._set_status(f"● 就绪 | {item.text()}", GREEN_OK)

    def _on_detect(self) -> None:
        """检测按钮 → 读取阈值 → 启动推理。"""
        if self._current_bgr is None:
            logger.warning("No image selected")
            return

        if not self._inference_manager.is_model_ready():
            self._set_status("● 加载模型中...", CYAN_INFO)

        conf = self._control_panel.get_conf_threshold()
        iou = self._control_panel.get_iou_threshold()
        self._control_panel.set_detect_enabled(False)
        self._set_status("● 推理中...", CYAN_INFO)
        self._inference_manager.detect(self._current_bgr, conf, iou)

    def _on_stop(self) -> None:
        """停止按钮 → 取消当前推理。"""
        self._inference_manager.stop()
        self._set_status("● 停止中...", YELLOW_WARNING)

    def _on_clear_boxes(self) -> None:
        """清除按钮 → 移除检测框 + 重置统计。"""
        self._image_viewer.clear_overlay()
        self._control_panel.reset_stats()

    def _on_result(self, detections: dict) -> None:
        """Manager 推理完成回调（主线程）→ 画框 + 统计。"""
        cancelled = detections.get("cancelled", False)
        t_ms = detections.get("inference_time_ms", 0.0)
        num = detections.get("num_detections", 0)

        if cancelled:
            self._image_viewer.clear_overlay()
            self._control_panel.reset_stats()
            self._set_status("● 已取消", YELLOW_WARNING)
        else:
            self._image_viewer.draw_boxes(detections)
            self._control_panel.update_result_counts(detections)
            color = RED_ALARM if num > 0 else GREEN_OK
            self._set_status(f"● 就绪 | {num} 缺陷 | {t_ms:.1f}ms", color)

        self._control_panel.set_detect_enabled(True)

    def _on_error(self, msg: str) -> None:
        """Manager 异常回调（主线程）→ 状态栏报错。"""
        logger.error(f"Detection error: {msg}")
        self._set_status(f"● 错误: {msg}", RED_ALARM)
        self._control_panel.set_detect_enabled(True)
