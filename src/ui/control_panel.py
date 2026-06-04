"""右侧控制面板。

属于 UI 层——提供检测控制按钮和参数滑块。
当前阶段所有操作仅记录日志，不触发任何算法。
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QSlider,
    QHBoxLayout,
    QGroupBox,
    QSizePolicy,
)

from src.config.app_config import CONF_THRESHOLD, IOU_THRESHOLD
from src.ui.theme import (
    BACKGROUND,
    SURFACE,
    BORDER,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    GREEN_OK,
    RED_ALARM,
    CYAN_INFO,
    FONT_FAMILY,
    FONT_SIZE_PANEL,
    FONT_WEIGHT_BOLD,
    SPACING_MD,
    SPACING_LG,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ControlPanel(QWidget):
    """右侧控制面板：检测按钮 + 参数滑块 + 统计信息占位。

    布局：
    ┌──────────────────────┐
    │  [检测] [停止] [清除] │
    │                      │
    │  置信度阈值 ────●────  │
    │  IoU 阈值   ────●────  │
    │                      │
    │  检测结果（占位）      │
    └──────────────────────┘
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setStyleSheet(self._panel_style())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING_MD, SPACING_MD, SPACING_MD, SPACING_MD)
        layout.setSpacing(SPACING_LG)

        # ── 标题 ──────────────────────────────────────────
        title = QLabel("CONTROL PANEL")
        title.setStyleSheet(f"""
            color: {TEXT_PRIMARY};
            font-family: {FONT_FAMILY};
            font-size: {FONT_SIZE_PANEL + 1}px;
            font-weight: {FONT_WEIGHT_BOLD};
        """)
        layout.addWidget(title)

        # ── 按钮组 ────────────────────────────────────────
        btn_group = QGroupBox("操作")
        btn_group.setStyleSheet(self._group_style())
        btn_layout = QVBoxLayout(btn_group)

        self._btn_detect = self._make_button("▶  检测", GREEN_OK)
        self._btn_detect.clicked.connect(self._on_detect)
        btn_layout.addWidget(self._btn_detect)

        self._btn_stop = self._make_button("■  停止", RED_ALARM)
        self._btn_stop.clicked.connect(self._on_stop)
        btn_layout.addWidget(self._btn_stop)

        self._btn_clear = self._make_button("✕  清除", CYAN_INFO)
        self._btn_clear.clicked.connect(self._on_clear)
        btn_layout.addWidget(self._btn_clear)

        layout.addWidget(btn_group)

        # ── 滑块组 ────────────────────────────────────────
        slider_group = QGroupBox("参数")
        slider_group.setStyleSheet(self._group_style())
        slider_layout = QVBoxLayout(slider_group)

        self._conf_slider, self._conf_label = self._make_slider(
            "置信度阈值", int(CONF_THRESHOLD * 100), 100
        )
        slider_layout.addLayout(self._conf_slider)

        self._iou_slider, self._iou_label = self._make_slider(
            "IoU 阈值", int(IOU_THRESHOLD * 100), 100
        )
        slider_layout.addLayout(self._iou_slider)

        layout.addWidget(slider_group)

        # ── 统计占位 ──────────────────────────────────────
        stats_group = QGroupBox("检测结果")
        stats_group.setStyleSheet(self._group_style())
        stats_layout = QVBoxLayout(stats_group)
        self._stats_label = QLabel("等待检测...")
        self._stats_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-family: {FONT_FAMILY};")
        stats_layout.addWidget(self._stats_label)
        layout.addWidget(stats_group)

        layout.addStretch()

    # ── 按钮回调（仅日志） ────────────────────────────────

    def _on_detect(self) -> None:
        logger.info("ControlPanel: 检测按钮点击（not implemented）")

    def _on_stop(self) -> None:
        logger.info("ControlPanel: 停止按钮点击（not implemented）")

    def _on_clear(self) -> None:
        logger.info("ControlPanel: 清除按钮点击（not implemented）")

    # ── 滑块回调（仅日志） ────────────────────────────────

    def _on_conf_changed(self, value: int) -> None:
        val = value / 100.0
        self._conf_label.setText(f"置信度阈值: {val:.2f}")
        logger.info(f"ControlPanel: 置信度阈值 → {val:.2f}")

    def _on_iou_changed(self, value: int) -> None:
        val = value / 100.0
        self._iou_label.setText(f"IoU 阈值: {val:.2f}")
        logger.info(f"ControlPanel: IoU 阈值 → {val:.2f}")

    # ── 组件工厂 ──────────────────────────────────────────

    def _make_button(self, text: str, color: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setMinimumHeight(36)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {SURFACE};
                color: {color};
                border: 1px solid {color};
                border-radius: 4px;
                font-family: {FONT_FAMILY};
                font-size: {FONT_SIZE_PANEL}px;
                font-weight: {FONT_WEIGHT_BOLD};
                padding: 6px 16px;
            }}
            QPushButton:hover {{
                background-color: {color};
                color: {BACKGROUND};
            }}
            QPushButton:pressed {{
                background-color: {color};
                color: {BACKGROUND};
            }}
        """)
        return btn

    def _make_slider(self, label_text: str, default_value: int, max_value: int):
        """创建滑块 + 标签组合布局。

        Returns:
            (QHBoxLayout, QLabel) — layout 可直接 addLayout
        """
        row = QHBoxLayout()

        lbl = QLabel(f"{label_text}: {default_value / 100:.2f}")
        lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-family: {FONT_FAMILY}; font-size: {FONT_SIZE_PANEL - 1}px;")
        lbl.setMinimumWidth(140)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, max_value)
        slider.setValue(default_value)
        slider.setStyleSheet(self._slider_style())
        slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        slider.setFixedHeight(24)

        if label_text == "置信度阈值":
            slider.valueChanged.connect(self._on_conf_changed)
        else:
            slider.valueChanged.connect(self._on_iou_changed)

        row.addWidget(lbl)
        row.addWidget(slider)

        # 保存 label 引用用于更新
        if label_text == "置信度阈值":
            self._conf_label = lbl
        else:
            self._iou_label = lbl

        return row, lbl

    # ── 样式 ──────────────────────────────────────────────

    @staticmethod
    def _panel_style() -> str:
        return f"""
            ControlPanel {{
                background-color: {BACKGROUND};
            }}
        """

    @staticmethod
    def _group_style() -> str:
        return f"""
            QGroupBox {{
                color: {TEXT_PRIMARY};
                font-family: {FONT_FAMILY};
                font-size: {FONT_SIZE_PANEL - 1}px;
                font-weight: {FONT_WEIGHT_BOLD};
                border: 1px solid {BORDER};
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 12px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }}
        """

    @staticmethod
    def _slider_style() -> str:
        return f"""
            QSlider::groove:horizontal {{
                background: {BORDER};
                height: 4px;
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {CYAN_INFO};
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {GREEN_OK};
            }}
            QSlider::add-page:horizontal {{
                background: {BORDER};
            }}
        """
