"""CSV 写入工具。

属于 Manager 层——纯 Python，零 Qt 依赖，在 Worker 线程安全调用。
Day 13 新增。
"""

import csv
from datetime import datetime
from pathlib import Path


def write_batch_csv(
    per_image_stats: list[dict],
    output_dir: Path,
    folder_name: str = "",
    suffix: str = "",
) -> Path:
    """将 per-image 统计列表写入 CSV 文件。

    CSV 字段:
        image_filename, num_detections, classes_detected, top_class,
        top_score, inference_time_ms, has_defect, error

    Args:
        per_image_stats: 每张图的统计 dict 列表
        output_dir: 输出目录
        folder_name: 文件夹名（用于文件命名，可选）
        suffix: 文件名后缀（如 "_cancelled"），可选

    Returns:
        写入的 CSV 文件路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{folder_name}" if folder_name else ""
    csv_path = output_dir / f"batch{tag}_{timestamp}{suffix}.csv"

    fieldnames = [
        "image_filename",
        "num_detections",
        "classes_detected",
        "top_class",
        "top_score",
        "inference_time_ms",
        "has_defect",
        "error",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for stat in per_image_stats:
            classes = stat.get("classes", [])
            scores = stat.get("scores", [])
            top_score = max(scores) if scores else 0.0
            top_idx = scores.index(top_score) if scores else -1
            top_class = classes[top_idx] if top_idx >= 0 else ""

            row = {
                "image_filename": stat.get("filename", "?"),
                "num_detections": stat.get("num_detections", 0),
                "classes_detected": ";".join(classes),
                "top_class": top_class,
                "top_score": round(top_score, 4),
                "inference_time_ms": round(stat.get("time_ms", 0.0), 1),
                "has_defect": stat.get("num_detections", 0) > 0,
                "error": stat.get("error", ""),
            }
            writer.writerow(row)

    return csv_path
