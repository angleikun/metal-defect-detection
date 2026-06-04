"""图像预览组件。

属于 UI 层——用于显示当前选中的图像。
基于 QGraphicsView + QGraphicsScene，为后续叠加检测框预留能力。
"""

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem

from src.config.app_config import IMAGE_DIR
from src.ui.theme import BACKGROUND, BORDER
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ImageViewer(QGraphicsView):
    """SCADA 风格图像预览画板。

    使用 QGraphicsView 而非 QLabel，为后续：
    - 缩放/平移（鼠标滚轮+拖拽）
    - 检测框叠加（QGraphicsRectItem）
    - 像素坐标映射
    预留架构空间。

    当前阶段仅加载一张占位图。
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # 深色背景
        self.setBackgroundBrush(Qt.GlobalColor.transparent)
        self.setStyleSheet(self._view_style())

        # 显示占位图
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._load_placeholder()

    # ── 占位图加载 ────────────────────────────────────────

    def _load_placeholder(self) -> None:
        """从 NEU-DET 数据集加载第一张可用图像作为占位图。"""
        img_path = self._find_first_image()
        if img_path is None:
            self._show_placeholder_text("No image found")
            return

        pixmap = QPixmap(str(img_path))
        if pixmap.isNull():
            self._show_placeholder_text("Failed to load image")
            return

        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._pixmap_item)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        logger.info(f"Loaded placeholder: {img_path.name}")

    def _find_first_image(self) -> Path | None:
        """查找 IMAGE_DIR 下任意一张 .jpg 文件。"""
        candidates = list(IMAGE_DIR.rglob("*.jpg"))
        if not candidates:
            candidates = list(IMAGE_DIR.rglob("*.png"))
        return candidates[0] if candidates else None

    def _show_placeholder_text(self, text: str) -> None:
        """无可用图像时显示提示文字。"""
        self._scene.addText(text)
        logger.warning(text)

    # ── 样式 ──────────────────────────────────────────────

    @staticmethod
    def _view_style() -> str:
        return f"""
            QGraphicsView {{
                background-color: {BACKGROUND};
                border: 1px solid {BORDER};
                border-radius: 2px;
            }}
        """
