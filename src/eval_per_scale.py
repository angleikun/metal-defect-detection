"""分尺度 mAP 评测 v2（按原图 200×200 坐标系，COCO 风格 AP 计算）。

用 sklearn average_precision_score 代替 11-point 插值，
对少样本分档也能给出合理结果。
"""

import sys
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
# AP 计算用 numpy 手动实现（避免依赖 sklearn）

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from utils.eval_protocol import EVAL_CONFIG

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "NEU-DET"
SPLITS = PROJECT_ROOT / "data" / "splits"
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

SMALL = 32 ** 2       # 1024
MEDIUM = 96 ** 2      # 9216


def load_test_gt():
    """加载 test set 所有 GT bbox，返回 {stem: [{cls, bbox, area, scale}]}."""
    test_stems = set((SPLITS / "test.txt").read_text().splitlines())
    gt_by_image = defaultdict(list)

    for split in ["train", "validation"]:
        ann_dir = DATA_DIR / split / "annotations"
        for xml_path in ann_dir.glob("*.xml"):
            stem = xml_path.stem
            if stem not in test_stems:
                continue
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall("object"):
                name = obj.find("name").text.strip()
                if name not in CLASSES:
                    continue
                bb = obj.find("bndbox")
                xmin = float(bb.find("xmin").text)
                ymin = float(bb.find("ymin").text)
                xmax = float(bb.find("xmax").text)
                ymax = float(bb.find("ymax").text)
                w, h = xmax - xmin, ymax - ymin
                area = w * h
                scale = "small" if area < SMALL else ("large" if area >= MEDIUM else "medium")
                gt_by_image[stem].append({
                    "cls": name, "bbox": [xmin, ymin, xmax, ymax], "area": area, "scale": scale,
                })
    return gt_by_image


def run_predictions():
    """用 EVAL_CONFIG 跑 test set 预测。"""
    from ultralytics import YOLO
    model = YOLO(str(PROJECT_ROOT / "models" / "exp4_crazing_focus" / "weights" / "best.pt"))
    results = model.predict(
        source=str(SPLITS / "test_paths.txt"),
        conf=EVAL_CONFIG["conf"], iou=EVAL_CONFIG["iou"],
        imgsz=EVAL_CONFIG["imgsz"], device=EVAL_CONFIG["device"],
        verbose=False, save=False,
    )
    preds_by_image = {}
    for r in results:
        stem = Path(r.path).stem
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            preds_by_image[stem] = []
            continue
        preds = []
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            preds.append({
                "cls": CLASSES[int(boxes.cls[i])],
                "conf": float(boxes.conf[i]),
                "bbox": [float(xyxy[0]) * 200 / 640, float(xyxy[1]) * 200 / 640,
                         float(xyxy[2]) * 200 / 640, float(xyxy[3]) * 200 / 640],
            })
        preds_by_image[stem] = preds
    return preds_by_image


def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-8)


def evaluate_per_scale(gt_dict, pred_dict):
    """按 S/M/L 分尺度计算 per-class mAP@0.5（COCO 风格）。"""
    scales = ["small", "medium", "large"]
    # 每个 scale 每个 class 收集 y_true 和 y_score
    data = {s: {c: {"y_true": [], "y_score": [], "gt_count": 0} for c in CLASSES} for s in scales}

    for stem, gts in gt_dict.items():
        preds = pred_dict.get(stem, [])

        # 对每个 GT，计数
        for gt in gts:
            data[gt["scale"]][gt["cls"]]["gt_count"] += 1

        # 匹配：pred -> GT（IoU >= 0.5 且类别一致）= TP，否则 = FP
        matched = set()
        for pred in sorted(preds, key=lambda x: x["conf"], reverse=True):
            best_iou, best_i = 0.0, -1
            for i, gt in enumerate(gts):
                if i in matched or pred["cls"] != gt["cls"]:
                    continue
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou, best_i = iou, i

            is_tp = best_iou >= 0.5
            pred_area = (pred["bbox"][2] - pred["bbox"][0]) * (pred["bbox"][3] - pred["bbox"][1])
            pred_scale = "small" if pred_area < SMALL else ("large" if pred_area >= MEDIUM else "medium")

            data[pred_scale][pred["cls"]]["y_true"].append(1 if is_tp else 0)
            data[pred_scale][pred["cls"]]["y_score"].append(pred["conf"])

            if is_tp:
                matched.add(best_i)

    # Compute AP per scale per class
    ap = {s: {} for s in scales}
    for s in scales:
        for cls in CLASSES:
            d = data[s][cls]
            if d["gt_count"] == 0:
                ap[s][cls] = float("nan")  # 无此类 GT
            elif len(d["y_true"]) == 0:
                ap[s][cls] = 0.0  # 有 GT 但无预测
            else:
                # Manual AP (COCO style: integral over PR curve)
                yt = np.array(d["y_true"]); ys = np.array(d["y_score"])
                order = np.argsort(-ys)
                yt = yt[order]
                tp_cum = np.cumsum(yt)
                fp_cum = np.cumsum(1 - yt)
                recalls = tp_cum / d["gt_count"]
                precisions = tp_cum / (tp_cum + fp_cum + 1e-8)
                # Interpolate precision: for each r, use max p at >= r
                for i in range(len(precisions) - 1, 0, -1):
                    precisions[i - 1] = max(precisions[i - 1], precisions[i])
                # AP = integral of interpolated precision over recall
                ap_val = 0.0
                prev_r = 0.0
                for r, p in zip(recalls, precisions):
                    ap_val += p * (r - prev_r)
                    prev_r = r
                ap[s][cls] = float(ap_val)

    return ap, data


def main():
    print("Loading test GT & predictions...")
    gt_dict = load_test_gt()
    pred_dict = run_predictions()

    # 统计 bbox 分布
    size_counts = {cls: {"small": 0, "medium": 0, "large": 0} for cls in CLASSES}
    for img_gts in gt_dict.values():
        for gt in img_gts:
            size_counts[gt["cls"]][gt["scale"]] += 1

    totals = {"small": 0, "medium": 0, "large": 0}
    for cls in CLASSES:
        for s in totals:
            totals[s] += size_counts[cls][s]
    gt_total = sum(totals.values())

    print(f"\nGT bbox distribution (test set, {gt_total} total):")
    print(f"  Small (<32^2):  {totals['small']:>4} ({totals['small']/gt_total*100:.1f}%)")
    print(f"  Medium:         {totals['medium']:>4} ({totals['medium']/gt_total*100:.1f}%)")
    print(f"  Large (>=96^2): {totals['large']:>4} ({totals['large']/gt_total*100:.1f}%)")

    print("\nComputing per-scale mAP...")
    ap, data = evaluate_per_scale(gt_dict, pred_dict)

    # ── 输出表格 ────────────────────────────────────────
    print()
    print("=" * 90)
    print("Per-Scale mAP@0.5 — Test Set (COCO Scale, 200x200 original coords)")
    print("=" * 90)
    print(f"{'Class':<20} {'small (<32^2)':>14} {'medium':>14} {'large (>=96^2)':>16} {'#GT S/M/L':>18}")
    print("-" * 90)

    valid_aps = {s: [] for s in ["small", "medium", "large"]}
    for cls in CLASSES:
        sv, mv, lv = ap["small"][cls], ap["medium"][cls], ap["large"][cls]
        gt_s, gt_m, gt_l = size_counts[cls]["small"], size_counts[cls]["medium"], size_counts[cls]["large"]

        s_str = "  N/A" if np.isnan(sv) else f"{sv:>6.4f}" + ("*" if gt_s < 30 else " ")
        m_str = "  N/A" if np.isnan(mv) else f"{mv:>6.4f}" + ("*" if gt_m < 30 else " ")
        l_str = "  N/A" if np.isnan(lv) else f"{lv:>6.4f}" + ("*" if gt_l < 30 else " ")

        print(f"{cls:<20} {s_str:>14} {m_str:>14} {l_str:>16} {f'{gt_s}/{gt_m}/{gt_l}':>18}")

        if not np.isnan(sv): valid_aps["small"].append(sv)
        if not np.isnan(mv): valid_aps["medium"].append(mv)
        if not np.isnan(lv): valid_aps["large"].append(lv)

    print("-" * 90)
    sa = np.mean(valid_aps["small"]) if valid_aps["small"] else float("nan")
    ma = np.mean(valid_aps["medium"]) if valid_aps["medium"] else float("nan")
    la = np.mean(valid_aps["large"]) if valid_aps["large"] else float("nan")
    print(f"{'Mean (excl N/A)':<20} {sa:>14.4f} {ma:>14.4f} {la:>16.4f}")
    print()
    print("  *  = < 30 GT bboxes in this scale, AP has high variance (for reference only)")
    print("  N/A = 0 GT bboxes in this scale, AP not computable")

    # ── 柱状图 ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(CLASSES)); w = 0.25
    for i, (scale, color, label) in enumerate([
        ("small", "#FF6B6B", "small (<32²)"),
        ("medium", "#FFD93D", "medium (32²-96²)"),
        ("large", "#6BCB77", "large (≥96²)"),
    ]):
        vals = [ap[scale][c] if not np.isnan(ap[scale][c]) else 0.0 for c in CLASSES]
        ax.bar(x + i * w, vals, w, color=color, label=label, alpha=0.85)

    ax.set_xticks(x + w); ax.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax.set_ylabel("mAP@0.5"); ax.set_title("Per-Scale mAP@0.5 on Test Set")
    ax.legend(fontsize=8); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig_path = PROJECT_ROOT / "results" / "test_size_breakdown.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nChart: {fig_path}")

    csv_path = PROJECT_ROOT / "results" / "test_size_breakdown.csv"
    with open(csv_path, "w") as f:
        f.write("class,scale,gt_count,mAP50,note\n")
        for cls in CLASSES:
            for s in ["small", "medium", "large"]:
                val = ap[s][cls]
                cnt = size_counts[cls][s]
                note = "N/A" if np.isnan(val) else ("low_sample" if cnt < 30 else "")
                f.write(f"{cls},{s},{cnt},{0 if np.isnan(val) else val:.4f},{note}\n")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
