"""主窗口骨架 —— SCADA 风格三栏布局。UI 层，只组合不写算法逻辑。
Day 12 增强：连接 InferenceManager ↔ UI 控件，端到端检测流程。"""

import cv2
import numpy as np

import webbrowser
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QSplitter,
    QListWidget, QListWidgetItem, QStatusBar, QLabel,
    QFileDialog, QMessageBox,
)

from src.config.app_config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, IMAGE_DIR, YOLO_BEST,
    PER_CLASS_CONF, BATCH_OUTPUT_DIR, REPORT_OUTPUT_DIR,
)
from src.manager.inference_manager import InferenceManager
from src.manager.batch_manager import BatchManager
from src.manager.report_manager import ReportManager
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
        self._batch_manager: BatchManager | None = None
        self._report_manager: ReportManager | None = None

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
        self._control_panel.batch_clicked.connect(self._on_batch_start)
        self._control_panel.report_clicked.connect(self._on_export_latest)

    def _set_status(self, text: str, color: str) -> None:
        """更新状态栏文字和颜色。"""
        self._status_label.setText(text)
        self._status_label.setStyleSheet(
            f"color: {color}; padding-left: {SPACING_MD}px;"
        )

    def _on_model_loaded(self, ok: bool) -> None:
        if ok:
            self._set_status("● 模型就绪", CYAN_INFO)
            self._init_batch_manager()
            self._init_report_manager()

    def _init_batch_manager(self) -> None:
        """模型加载后创建 BatchManager（共享 detector 引用）。"""
        if self._batch_manager is not None:
            return
        detector = self._inference_manager._detector
        self._batch_manager = BatchManager(detector, self)
        self._batch_manager.batch_progress.connect(self._on_batch_progress)
        self._batch_manager.batch_done.connect(self._on_batch_done)
        self._batch_manager.batch_cancelled.connect(self._on_batch_cancelled)
        self._batch_manager.batch_error.connect(self._on_batch_error)

    def _init_report_manager(self) -> None:
        """创建 ReportManager（零开销，无延迟加载）。"""
        if self._report_manager is not None:
            return
        self._report_manager = ReportManager(self)
        self._report_manager.report_ready.connect(self._on_report_finished)
        self._report_manager.report_error.connect(self._on_report_error)

    # ── 菜单栏 ──────────────────────────────────────────────

    def _build_menu_bar(self) -> None:
        mb = self.menuBar()
        mb.setStyleSheet(_MENU_STYLE)

        # ── 文件 ────────────────────────────────────────────
        file_menu = mb.addMenu("文件")
        file_menu.addAction(self._act("打开图像...", "Ctrl+O", self._on_open_image))
        file_menu.addAction(self._act("打开文件夹...", "Ctrl+Shift+O", self._on_open_folder))
        file_menu.addSeparator()
        file_menu.addAction(self._act("导出报表...", "Ctrl+E", self._on_export_report))
        file_menu.addSeparator()
        file_menu.addAction(self._act("退出", "Alt+F4", self.close))

        # ── 配置 ────────────────────────────────────────────
        cfg_menu = mb.addMenu("配置")
        cfg_menu.addAction(self._act("模型信息...", None, self._on_model_info))
        cfg_menu.addAction(self._act("Per-Class 阈值...", None, self._on_threshold_info))

        # ── 视图 ────────────────────────────────────────────
        view_menu = mb.addMenu("视图")
        view_menu.addAction(self._act("重置布局", None, self._on_reset_layout))
        view_menu.addAction(self._act("清空日志", None, self._on_clear_log))

        # ── 报警 ────────────────────────────────────────────
        alarm_menu = mb.addMenu("报警")
        alarm_menu.addAction(self._act("报警历史", None, self._on_alarm_history))
        self._alarm_mute_action = self._act("静音报警", None, self._on_alarm_mute)
        self._alarm_mute_action.setCheckable(True)
        alarm_menu.addAction(self._alarm_mute_action)

        # ── 帮助 ────────────────────────────────────────────
        help_menu = mb.addMenu("帮助")
        help_menu.addAction(self._act("关于...", None, self._on_about))
        help_menu.addAction(self._act("GitHub", None, self._on_github))

    def _on_clear_log(self) -> None:
        if self._log_panel:
            self._log_panel.clear()

    def _act(self, text: str, shortcut: str | None, slot) -> QAction:
        action = QAction(text, self)  # parent=self fixes Windows menu invisibility
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        return action

    # ── 菜单回调 ────────────────────────────────────────────

    def _on_open_image(self) -> None:
        """文件 → 打开图像：QFileDialog 选单张图。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "打开图像", "",
            "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if path:
            self._current_bgr = cv2.imread(path)
            if self._current_bgr is None:
                QMessageBox.warning(self, "错误", f"无法读取图像:\n{path}")
                return
            self._image_viewer.load_image(path)
            self._control_panel.reset_stats()
            self._set_status(f"● 已打开: {Path(path).name}", GREEN_OK)
            logger.info(f"打开图像: {path}")

    def _on_model_info(self) -> None:
        """配置 → 模型信息：弹窗显示当前模型状态。"""
        ready = self._inference_manager.is_model_ready()
        if ready:
            det = self._inference_manager._detector
            info = (
                f"模型路径: {det._model_path}\n"
                f"推理设备: {det.device}\n"
                f"ONNX 模型: models/exp4_crazing_focus/weights/best.onnx\n"
                f"ONNX Runtime: 1.23.2 (CPU)"
            )
        else:
            info = "模型尚未加载（首次 detect 时自动加载）"
        QMessageBox.information(self, "模型信息", info)

    def _on_threshold_info(self) -> None:
        """配置 → Per-Class 阈值：弹窗显示 6 类置信度阈值。"""
        from src.config.app_config import PER_CLASS_CONF
        lines = ["Per-Class Confidence Thresholds:\n"]
        for cls_name, th in PER_CLASS_CONF.items():
            lines.append(f"  {cls_name:<20s}  {th:.2f}")
        QMessageBox.information(self, "Per-Class 置信度阈值", "\n".join(lines))

    def _on_reset_layout(self) -> None:
        """视图 → 重置布局：恢复三栏默认比例 2:5:3。"""
        splitter = self.findChild(QSplitter)
        if splitter:
            splitter.setStretchFactor(0, 2)
            splitter.setStretchFactor(1, 5)
            splitter.setStretchFactor(2, 3)
            self._set_status("● 布局已重置 (2:5:3)", CYAN_INFO)
            logger.info("布局已重置")

    def _on_alarm_history(self) -> None:
        """报警 → 报警历史：弹窗显示信息（报警系统未实现）。"""
        QMessageBox.information(
            self, "报警历史",
            "报警模块尚未实现。\n\n"
            "计划功能：\n"
            "  - 缺陷检出即报警（红色状态栏闪烁）\n"
            "  - 连续 N 张检出同类缺陷 → 升级报警\n"
            "  - 报警历史 CSV 导出\n\n"
            "当前阶段可通过批处理 CSV 查看缺陷检出记录。"
        )

    def _on_alarm_mute(self) -> None:
        """报警 → 静音报警：切换静音状态。"""
        muted = self._alarm_mute_action.isChecked()
        if muted:
            self._set_status("● 报警已静音", YELLOW_WARNING)
            logger.info("报警已静音")
        else:
            self._set_status("● 报警已恢复", GREEN_OK)
            logger.info("报警已恢复")

    def _on_about(self) -> None:
        """帮助 → 关于：弹窗显示版本信息。"""
        QMessageBox.about(
            self, "关于 — Metal Defect Detection",
            "<h3>Metal Defect Detection v1.0</h3>"
            "<p>工业金属表面缺陷检测系统</p>"
            "<p>NEU-DET 6 类 / YOLOv8n / PyQt6 SCADA</p>"
            "<hr>"
            "<p><b>技术栈:</b> PyTorch 2.5.1 · ultralytics 8.4.60 · "
            "PyQt6 6.6.1 · ONNX Runtime 1.23.2 · OpenCV 4.13</p>"
            "<p><b>作者:</b> linsanqin</p>"
            "<p><b>GitHub:</b> <a href='https://github.com/angleikun'>"
            "github.com/angleikun</a></p>"
        )

    def _on_github(self) -> None:
        """帮助 → GitHub：在浏览器打开仓库。"""
        webbrowser.open("https://github.com/angleikun")

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
        """停止按钮 → 取消当前推理/批处理。"""
        if self._batch_manager and self._batch_manager.is_busy():
            self._batch_manager.stop()
        else:
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

    # ── 批处理事件（Day 13） ──────────────────────────────

    def _on_open_folder(self) -> None:
        """打开文件夹对话框 → 启动批处理。"""
        dlg = QFileDialog(self, "选择图像文件夹")
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOption(QFileDialog.Option.ShowDirsOnly, True)
        if dlg.exec() == QFileDialog.DialogCode.Accepted:
            folder = dlg.selectedFiles()[0]
            self._start_batch(folder)

    def _on_batch_start(self) -> None:
        """批处理按钮 → 打开文件夹对话框。"""
        self._on_open_folder()

    def _start_batch(self, folder_path: str,
                     extensions: tuple[str, ...] = (".jpg", ".jpeg")) -> None:
        """启动批处理。"""
        if self._batch_manager is None:
            self._set_status("● 模型未加载", RED_ALARM)
            return
        if self._batch_manager.is_busy():
            logger.warning("Batch already in progress")
            return

        conf = min(PER_CLASS_CONF.values())
        iou = self._control_panel.get_iou_threshold()
        self._control_panel.set_batch_mode(True)
        self._set_status(f"● 批处理中: {folder_path}", CYAN_INFO)
        logger.info(f"批处理开始: 文件夹={folder_path}")
        self._batch_manager.run_batch(
            folder_path, conf, iou,
            per_class_conf=PER_CLASS_CONF, extensions=extensions,
        )

    def _on_batch_progress(self, completed: int, total: int, filename: str) -> None:
        """批处理进度 → 进度条 + 状态栏。"""
        self._control_panel.update_batch_progress(completed, total, filename)
        if completed % 50 == 0 or completed == total:
            pct = int(completed / total * 100)
            self._set_status(f"● 批处理: {completed}/{total} ({pct}%)", CYAN_INFO)

    def _on_batch_done(self, summary: dict) -> None:
        """批处理完成。"""
        t = summary["total"]
        d = summary["defect_count"]
        avg = summary["avg_time_ms"]
        csv_path = summary["csv_path"]
        self._batch_csv_path = csv_path  # 保存供报表导出使用
        self._control_panel.set_batch_mode(False)
        self._control_panel.set_report_enabled(True)  # Day 14: 激活导出按钮
        self._set_status(f"● 批处理完成: {d}/{t} 有缺陷, avg {avg:.1f}ms", GREEN_OK)
        logger.info(f"批处理完成: {d}/{t} 张有缺陷, avg {avg:.1f}ms, csv={csv_path}")

    def _on_batch_cancelled(self, count: int, csv_path: str) -> None:
        """批处理被取消。"""
        self._control_panel.set_batch_mode(False)
        self._set_status(f"● 已取消 (已处理 {count} 张)", YELLOW_WARNING)
        if csv_path:
            logger.info(f"批处理已取消 (已处理 {count} 张)，部分结果: {csv_path}")
        else:
            logger.info(f"批处理已取消 (已处理 {count} 张)")

    def _on_batch_error(self, msg: str) -> None:
        """批处理异常。"""
        self._control_panel.set_batch_mode(False)
        self._set_status(f"● 批处理错误: {msg}", RED_ALARM)
        logger.error(f"批处理错误: {msg}")

    # ── 报表导出事件（Day 14） ─────────────────────────────

    def _on_export_report(self) -> None:
        """菜单导出 → 弹出 QFileDialog 选 CSV → 生成报表。"""
        dlg = QFileDialog(self, "选择批处理 CSV 文件")
        dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
        dlg.setNameFilter("CSV 文件 (*.csv)")
        if dlg.exec() == QFileDialog.DialogCode.Accepted:
            csv_path = dlg.selectedFiles()[0]
            self._generate_report(csv_path)

    def _on_export_latest(self) -> None:
        """控制面板按钮 → 自动选最新 batch CSV → 生成报表。"""
        csv_path = getattr(self, "_batch_csv_path", None)
        if not csv_path:
            # 尝试找最新的 batch CSV
            import os
            csv_dir = BATCH_OUTPUT_DIR
            if csv_dir.is_dir():
                csv_files = sorted(
                    csv_dir.glob("batch_*.csv"),
                    key=os.path.getmtime, reverse=True,
                )
                non_cancelled = [f for f in csv_files if "_cancelled" not in f.name]
                if non_cancelled:
                    csv_path = str(non_cancelled[0])

        if not csv_path:
            logger.warning("No batch CSV found for report")
            self._set_status("● 无可用 CSV，请先运行批处理", YELLOW_WARNING)
            return

        self._generate_report(csv_path)

    def _generate_report(self, csv_path: str) -> None:
        """启动报表生成。"""
        if self._report_manager is None:
            self._init_report_manager()
        logger.info(f"导出报表: {csv_path}")
        self._set_status("● 生成报表中...", CYAN_INFO)
        self._report_manager.generate_html_report(csv_path)

    def _on_report_finished(self, html_path: str) -> None:
        """报表生成完成 → 自动打开浏览器。"""
        self._set_status(f"● 报表已生成: {html_path}", GREEN_OK)
        logger.info(f"报表已生成: {html_path}")
        webbrowser.open(f"file:///{html_path}")

    def _on_report_error(self, msg: str) -> None:
        """报表生成异常。"""
        self._set_status(f"● 报表错误: {msg}", RED_ALARM)
        logger.error(f"报表错误: {msg}")
