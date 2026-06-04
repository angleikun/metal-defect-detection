"""报表图表生成。

属于 Algorithm 层——用 matplotlib 生成图表，返回 base64 字符串。
Agg 后端避免与 PyQt6 冲突，零 Qt 依赖。
Day 14 新增。
"""

import base64
import csv
import io
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 必须在 import pyplot 之前！
import matplotlib.pyplot as plt
import numpy as np

from src.config.app_config import CLASS_NAMES, CLASS_COLORS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── 全局 matplotlib 样式：白色背景，适合打印 ──────────────
plt.rcParams.update({
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": "#1f2328",
    "axes.labelcolor": "#1f2328",
    "text.color": "#1f2328",
    "xtick.color": "#1f2328",
    "ytick.color": "#1f2328",
    "grid.color": "#e1e4e8",
    "font.family": "sans-serif",
    "font.size": 10,
})


def _read_csv(csv_path: str) -> list[dict]:
    """读取 CSV 文件，返回每行 dict 列表。"""
    rows = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def generate_class_distribution_chart(csv_path: str) -> str:
    """生成类别检出分布饼图 → base64 PNG 字符串。

    统计每类在所有 images 中出现过几次（classes_detected 包含该类即算一次）。

    Args:
        csv_path: 批处理 CSV 文件路径

    Returns:
        base64 编码的 PNG 图片字符串
    """
    rows = _read_csv(csv_path)
    class_count = defaultdict(int)

    for r in rows:
        classes = r.get("classes_detected", "")
        if classes:
            for cls_name in classes.split(";"):
                if cls_name in CLASS_NAMES:
                    class_count[cls_name] += 1
        # 无检出也算 "No Defect" 吗？不，只统计有检出的。

    if not class_count:
        logger.warning("No detections found in CSV, pie chart will be empty")

    labels = list(class_count.keys())
    values = list(class_count.values())
    colors = [
        CLASS_COLORS[CLASS_NAMES.index(lb)] if lb in CLASS_NAMES else "#cccccc"
        for lb in labels
    ]

    total_boxes = sum(values)
    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        values, labels=None, autopct="%1.1f%%",
        startangle=140, colors=colors, pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(9)
    # Day 15 polish: legend shows class names only (no detection counts)
    ax.legend(wedges, labels,
              title="Defect Classes", loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))
    ax.set_title(f"Class Distribution by Detection Count "
                 f"(n={total_boxes} boxes total)",
                 fontweight="bold", fontsize=13)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def generate_time_distribution_chart(csv_path: str) -> str:
    """生成推理耗时分布直方图 → base64 PNG 字符串。

    Args:
        csv_path: 批处理 CSV 文件路径

    Returns:
        base64 编码的 PNG 图片字符串
    """
    rows = _read_csv(csv_path)
    times = [float(r["inference_time_ms"]) for r in rows
             if r.get("inference_time_ms") and not r.get("error")]

    if not times:
        logger.warning("No valid times in CSV")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(times, bins=40, color="#0969da", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Inference Time (ms)")
    ax.set_ylabel("Image Count")
    ax.set_title(f"Inference Time Distribution (n={len(times)}, "
                 f"avg={np.mean(times):.1f}ms, "
                 f"median={np.median(times):.1f}ms)",
                 fontweight="bold", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    # 标注均值和 P99
    mean_t = np.mean(times)
    p99_t = np.percentile(times, 99)
    ax.axvline(mean_t, color="#ff1744", linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_t:.1f}ms")
    ax.axvline(p99_t, color="#ffd600", linestyle="--", linewidth=1.5,
               label=f"P99: {p99_t:.1f}ms")
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def generate_confidence_histogram(csv_path: str) -> str:
    """生成各类别 top_score 置信度分布直方图 → base64 PNG 字符串。

    Args:
        csv_path: 批处理 CSV 文件路径

    Returns:
        base64 编码的 PNG 图片字符串
    """
    rows = _read_csv(csv_path)
    # 按类别分组 scores
    class_scores = defaultdict(list)
    for r in rows:
        cls_name = r.get("top_class", "").strip()
        score_str = r.get("top_score", "0")
        if cls_name and score_str and cls_name in CLASS_NAMES:
            try:
                class_scores[cls_name].append(float(score_str))
            except ValueError:
                pass

    if not class_scores:
        logger.warning("No valid scores in CSV")

    fig, ax = plt.subplots(figsize=(8, 5))
    for cls_name in CLASS_NAMES:
        scores = class_scores.get(cls_name, [])
        if scores:
            color = CLASS_COLORS[CLASS_NAMES.index(cls_name)]
            ax.hist(scores, bins=30, alpha=0.5, label=f"{cls_name} (n={len(scores)})",
                    color=color, edgecolor="white")

    ax.set_xlabel("Top Confidence Score")
    ax.set_ylabel("Image Count")
    ax.set_title("Per-Class Confidence Score Distribution",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")
