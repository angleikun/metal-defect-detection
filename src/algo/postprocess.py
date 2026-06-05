"""后处理工具函数。

属于 Algorithm 层——纯算法，零 Qt 依赖。
将 ultralytics 原始输出转换为 UI 可直接使用的 dict。

Day 13: 新增 per-class threshold 过滤 (_filter_by_per_class)。
"""

import numpy as np

from src.config.app_config import CLASS_NAMES, CLASS_COLORS


def get_box_color(class_id: int) -> str:
    """返回指定类别的 BBox 渲染色（十六进制）。

    Args:
        class_id: 类别索引 (0-5)

    Returns:
        "#RRGGBB" 颜色字符串，越界返回白色
    """
    if 0 <= class_id < len(CLASS_COLORS):
        return CLASS_COLORS[class_id]
    return "#FFFFFF"


def get_class_name(class_id: int) -> str:
    """返回指定类别的名称。

    Args:
        class_id: 类别索引 (0-5)

    Returns:
        类名字符串，越界返回 "unknown"
    """
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return "unknown"


def extract_detections(
    ultralytics_result,
    image_shape: tuple[int, int],
    inference_time_ms: float,
    cancelled: bool = False,
    per_class_conf: dict | None = None,
) -> dict:
    """从 ultralytics.Results 提取结构化检测结果。

    Args:
        ultralytics_result: ultralytics.engine.results.Results 对象
        image_shape: 原始图像 (height, width)
        inference_time_ms: 推理耗时（毫秒）
        cancelled: 是否被取消
        per_class_conf: 可选，如 {"crazing": 0.05, ...}，按类别阈值过滤 box。
                        None 时不启用过滤（Day 12 行为）。

    Returns:
        dict with keys: boxes, classes, scores, class_names,
                        inference_time_ms, num_detections, image_shape, cancelled
    """
    if cancelled or ultralytics_result is None or ultralytics_result.boxes is None:
        return _empty_result(image_shape, inference_time_ms, cancelled)

    boxes_data = ultralytics_result.boxes

    # xyxy: [N, 4], cls: [N, 1], conf: [N, 1]
    xyxy = boxes_data.xyxy.cpu().numpy() if hasattr(boxes_data.xyxy, "cpu") else np.array(boxes_data.xyxy)
    cls_ids = boxes_data.cls.cpu().numpy() if hasattr(boxes_data.cls, "cpu") else np.array(boxes_data.cls)
    confs = boxes_data.conf.cpu().numpy() if hasattr(boxes_data.conf, "cpu") else np.array(boxes_data.conf)

    boxes = xyxy.astype(float).tolist() if len(xyxy) > 0 else []
    classes = cls_ids.astype(int).tolist() if len(cls_ids) > 0 else []
    scores = confs.astype(float).tolist() if len(confs) > 0 else []
    names = [get_class_name(c) for c in classes]

    # Day 13: per-class threshold 过滤
    if per_class_conf is not None and len(boxes) > 0:
        boxes, classes, scores, names = _filter_by_per_class(
            boxes, classes, scores, names, per_class_conf
        )

    return {
        "boxes": boxes,
        "classes": classes,
        "scores": scores,
        "class_names": names,
        "inference_time_ms": round(inference_time_ms, 2),
        "num_detections": len(boxes),
        "image_shape": image_shape,
        "cancelled": cancelled,
    }


def _filter_by_per_class(
    boxes: list,
    classes: list,
    scores: list,
    class_names: list,
    per_class_conf: dict,
) -> tuple[list, list, list, list]:
    """按 per-class 阈值过滤检测框。

    只保留 score >= per_class_conf.get(class_name, 0.25) 的 box。
    """
    kept = []
    for i, name in enumerate(class_names):
        threshold = per_class_conf.get(name, 0.25)
        if scores[i] >= threshold:
            kept.append(i)

    if len(kept) == len(boxes):
        return boxes, classes, scores, class_names

    return (
        [boxes[i] for i in kept],
        [classes[i] for i in kept],
        [scores[i] for i in kept],
        [class_names[i] for i in kept],
    )


def _empty_result(
    image_shape: tuple[int, int], inference_time_ms: float, cancelled: bool
) -> dict:
    """返回空检测结果。"""
    return {
        "boxes": [],
        "classes": [],
        "scores": [],
        "class_names": [],
        "inference_time_ms": round(inference_time_ms, 2),
        "num_detections": 0,
        "image_shape": image_shape,
        "cancelled": cancelled,
    }
