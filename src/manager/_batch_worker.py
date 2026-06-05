"""批量检测 Worker。

属于 Manager 层——QObject 子类，运行在子线程。
Day 13 新增。批量遍历图像文件夹，每张调用 YoloDetector，
emit progress 每 10 张，支持取消，批处理完成后写 CSV。
"""

import cv2
import traceback
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from src.algo.postprocess import extract_detections
from src.algo.yolo_detector import YoloDetector
from src.config.app_config import BATCH_EMIT_EVERY, BATCH_OUTPUT_DIR
from src.manager._csv_writer import write_batch_csv
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BatchWorker(QObject):
    """批量推理 Worker。QObject 子类，moveToThread 搬移到子线程。

    信号：
        progress(int, int, str):  (completed, total, current_filename)
        batch_finished(dict):      汇总 dict + csv_path
        cancelled(int):            已处理张数
        error(str, str):           (错误消息, traceback)
    """

    progress = pyqtSignal(int, int, str)
    batch_finished = pyqtSignal(dict)
    cancelled = pyqtSignal(int, str)  # Day 13 fix: (count, csv_path)
    error = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancel_flag = False
        self._detector: YoloDetector | None = None

    def run(
        self,
        image_paths: list[str],
        conf: float,
        iou: float,
        per_class_conf: dict | None = None,
    ) -> None:
        """批量推理入口。由 thread.started 信号触发。

        Args:
            image_paths: 图像文件路径列表
            conf: YOLO 推理 conf 阈值
            iou: NMS IoU 阈值
            per_class_conf: per-class 后处理过滤阈值
        """
        total = len(image_paths)
        per_image: list[dict] = []
        total_time_ms = 0.0
        defect_count = 0

        logger.info(f"BatchWorker: starting {total} images")

        for i, path_str in enumerate(image_paths):
            # ★ 取消点：每张图开始前检查
            if self._cancel_flag:
                logger.info(f"BatchWorker: cancelled after {i} images")
                # 写入部分 CSV（已处理的 N 张）
                try:
                    csv_path = write_batch_csv(
                        per_image, BATCH_OUTPUT_DIR,
                        Path(image_paths[0]).parent.name if image_paths else "",
                        suffix="_cancelled",
                    )
                except Exception:
                    csv_path = None
                self.cancelled.emit(i, str(csv_path) if csv_path else "")
                return

            path = Path(path_str)
            stat = {"filename": path.name,
                    "num_detections": 0, "classes": [], "scores": [],
                    "time_ms": 0.0, "error": ""}

            try:
                img = cv2.imread(str(path))
                if img is None:
                    stat["error"] = "read_failed"
                    per_image.append(stat)
                    continue

                if self._detector is None:
                    self.error.emit("detector not set", "")
                    return

                raw_result, t_ms = self._detector.predict(
                    img, conf=conf, iou=iou
                )
                detections = extract_detections(
                    raw_result, img.shape[:2], t_ms,
                    per_class_conf=per_class_conf,
                )

                stat["num_detections"] = detections["num_detections"]
                stat["classes"] = detections["class_names"]
                stat["scores"] = [round(s, 4) for s in detections["scores"]]
                stat["time_ms"] = t_ms
                total_time_ms += t_ms
                if detections["num_detections"] > 0:
                    defect_count += 1

            except Exception as exc:
                stat["error"] = str(exc)
                logger.debug(f"BatchWorker: error on {path.name}: {exc}")

            per_image.append(stat)

            # ★ EMIT_EVERY：每 10 张 emit 一次进度
            if (i + 1) % BATCH_EMIT_EVERY == 0 or (i + 1) == total:
                self.progress.emit(i + 1, total, path.name)

        # ── 汇总 ────────────────────────────────────────────
        avg_ms = total_time_ms / total if total > 0 else 0.0
        folder_name = Path(image_paths[0]).parent.name if image_paths else ""

        # Worker 线程写 CSV
        try:
            csv_path = write_batch_csv(per_image, BATCH_OUTPUT_DIR, folder_name)
        except Exception as exc:
            self.error.emit(f"CSV write failed: {exc}", traceback.format_exc())
            return

        summary = {
            "total": total,
            "defect_count": defect_count,
            "total_time_ms": round(total_time_ms, 1),
            "avg_time_ms": round(avg_ms, 1),
            "csv_path": str(csv_path),
            "per_image": per_image,
        }
        logger.info(
            f"BatchWorker: done — {defect_count}/{total} have defects, "
            f"avg {avg_ms:.1f}ms/img, csv={csv_path.name}"
        )
        self.batch_finished.emit(summary)

    def cancel(self) -> None:
        """设置取消标志（主线程调用，线程安全）。"""
        self._cancel_flag = True
        logger.debug("BatchWorker: cancel flag set")
