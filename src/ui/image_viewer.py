"""图像预览组件。

属于 UI 层——显示图像 + 叠加检测框。
基于 QGraphicsView + QGraphicsScene，用 QGraphicsItem 画框。
"""

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QColor, QPen, QBrush, QFont
from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsSimpleTextItem,
)

from src.config.app_config import IMAGE_DIR, DETECTION_LINE_WIDTH
from src.ui.theme import BACKGROUND, BORDER, FONT_FAMILY
from src.algo.postprocess import get_box_color, get_class_name
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ImageViewer(QGraphicsView):
    """SCADA 风格图像预览 + 检测框叠加。

    Day 12 增强：
    - load_image(): 动态加载指定图像
    - draw_boxes(): 叠加彩色检测框 + 类名 + 置信度
    - clear_overlay(): 清除框
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setBackgroundBrush(Qt.GlobalColor.transparent)
        self.setStyleSheet(self._view_style())

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._overlay_items: list = []   # QGraphicsItem 列表
        self._current_pixmap: QPixmap | None = None
        self._current_image_path: Path | None = None

        self._load_placeholder()

    # ── 占位图 ──────────────────────────────────────────────

    def _load_placeholder(self) -> None:
        img_path = self._find_first_image()
        if img_path is None:
            self._show_placeholder_text("No image found")
            return
        self.load_image(img_path)

    def _find_first_image(self) -> Path | None:
        candidates = list(IMAGE_DIR.rglob("*.jpg"))
        if not candidates:
            candidates = list(IMAGE_DIR.rglob("*.png"))
        return candidates[0] if candidates else None

    def _show_placeholder_text(self, text: str) -> None:
        self._scene.addText(text)
        logger.warning(text)

    # ── 图像加载 ────────────────────────────────────────────

    def load_image(self, image_path: str | Path) -> None:
        """加载指定图像，替换当前显示。

        Args:
            image_path: 图像文件路径
        """
        image_path = Path(image_path)
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            logger.warning(f"Failed to load: {image_path.name}")
            return

        # 清除旧内容
        self.clear_overlay()
        self._scene.clear()

        # 加载新图像
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._pixmap_item)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

        self._current_pixmap = pixmap
        self._current_image_path = image_path
        self._overlay_items = []
        logger.debug(f"Loaded: {image_path.name} ({pixmap.width()}×{pixmap.height()})")

    # ── 检测框叠加 ──────────────────────────────────────────

    def draw_boxes(self, detections: dict) -> None:
        """根据检测结果画框 + 标签。

        Args:
            detections: extract_detections() 返回的 dict
        """
        self.clear_overlay()

        boxes = detections.get("boxes", [])
        classes = detections.get("classes", [])
        scores = detections.get("scores", [])

        font = QFont(FONT_FAMILY, 10)
        font.setBold(True)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            class_id = classes[i] if i < len(classes) else 0
            score = scores[i] if i < len(scores) else 0.0
            color_hex = get_box_color(class_id)
            label = f"{get_class_name(class_id)} {score:.2f}"

            # 矩形框
            pen = QPen(QColor(color_hex))
            pen.setWidth(DETECTION_LINE_WIDTH)
            pen.setStyle(Qt.PenStyle.SolidLine)
            rect = QGraphicsRectItem(x1, y1, w, h)
            rect.setPen(pen)
            rect.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            self._scene.addItem(rect)
            self._overlay_items.append(rect)

            # 文字标签（框左上角上方）
            text = QGraphicsSimpleTextItem(label)
            text.setFont(font)
            text.setBrush(QBrush(QColor(color_hex)))
            text.setPos(x1, max(0, y1 - 16))
            self._scene.addItem(text)
            self._overlay_items.append(text)

        logger.debug(f"draw_boxes: {len(boxes)} boxes rendered")

    def clear_overlay(self) -> None:
        """移除所有检测框和标签。"""
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items.clear()

    # ── 属性 ────────────────────────────────────────────────

    @property
    def current_image_path(self) -> Path | None:
        """当前显示的图像路径。"""
        return self._current_image_path

    @property
    def current_pixmap(self) -> QPixmap | None:
        """当前显示的 QPixmap。"""
        return self._current_pixmap

    # ── 样式 ────────────────────────────────────────────────

    @staticmethod
    def _view_style() -> str:
        return f"""
            QGraphicsView {{
                background-color: {BACKGROUND};
                border: 1px solid {BORDER};
                border-radius: 2px;
            }}
        """
