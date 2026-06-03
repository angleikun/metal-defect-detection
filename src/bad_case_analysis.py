"""Bad case 分析：test set 上找最难 10 个样本 + 可视化。

最难定义：漏检 / 误检 / 类别错 / 低 IoU。
优先 crazing 和 rolled-in_scale。
"""

import sys
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from utils.eval_protocol import EVAL_CONFIG

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "NEU-DET"
SPLITS = PROJECT_ROOT / "data" / "splits"
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
OUT_DIR = PROJECT_ROOT / "results" / "bad_cases"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_COLORS = {
    "crazing": "#E74C3C", "inclusion": "#3498DB", "patches": "#2ECC71",
    "pitted_surface": "#F39C12", "rolled-in_scale": "#9B59B6", "scratches": "#1ABC9C",
}


def load_all_data():
    """加载 test set GT + 预测。"""
    test_stems = set((SPLITS / "test.txt").read_text().splitlines())
    gt_dict = {}
    img_paths = {}

    for split in ["train", "validation"]:
        ann_dir = DATA_DIR / split / "annotations"
        for cls in CLASSES:
            img_dir = DATA_DIR / split / "images" / cls
            if not img_dir.is_dir():
                continue
            for img_path in img_dir.glob("*.jpg"):
                stem = img_path.stem
                if stem not in test_stems:
                    continue
                img_paths[stem] = img_path
                xml_path = ann_dir / f"{stem}.xml"
                if xml_path.exists():
                    tree = ET.parse(xml_path); root = tree.getroot()
                    gts = []
                    for obj in root.findall("object"):
                        name = obj.find("name").text.strip()
                        bb = obj.find("bndbox")
                        gts.append({
                            "cls": name,
                            "bbox": [float(bb.find("xmin").text), float(bb.find("ymin").text),
                                     float(bb.find("xmax").text), float(bb.find("ymax").text)],
                        })
                    gt_dict[stem] = gts
                else:
                    gt_dict[stem] = []

    # 跑预测
    from ultralytics import YOLO
    model = YOLO(str(PROJECT_ROOT / "models" / "exp4_crazing_focus" / "weights" / "best.pt"))
    preds_dict = {}
    for stem, img_path in img_paths.items():
        r = model.predict(
            str(img_path), conf=EVAL_CONFIG["conf"], iou=EVAL_CONFIG["iou"],
            imgsz=EVAL_CONFIG["imgsz"], device=EVAL_CONFIG["device"],
            verbose=False, save=False,
        )
        boxes = r[0].boxes
        if boxes is None or len(boxes) == 0:
            preds_dict[stem] = []
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
        preds_dict[stem] = preds

    return gt_dict, preds_dict, img_paths


def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-8)


def analyze_errors(gt_dict, preds_dict):
    """分析每个样本的错误类型，生成 bad cases 列表。"""
    bad_cases = []

    for stem, gts in gt_dict.items():
        preds = preds_dict.get(stem, [])
        matched_gt = set()
        matched_pred = set()

        # 贪心匹配 (IoU >= 0.5)
        for pi, pred in enumerate(preds):
            best_iou, best_gi = 0.0, -1
            for gi, gt in enumerate(gts):
                if gi in matched_gt:
                    continue
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= 0.5:
                matched_gt.add(best_gi)
                matched_pred.add(pi)
                # 类别错
                if pred["cls"] != gts[best_gi]["cls"]:
                    bad_cases.append({
                        "stem": stem, "gts": gts, "preds": preds,
                        "error_type": "class_error",
                        "detail": f"pred={pred['cls']}, gt={gts[best_gi]['cls']}",
                        "iou": best_iou, "score": 0.3,
                    })

        # 漏检（GT 无匹配）
        for gi, gt in enumerate(gts):
            if gi not in matched_gt:
                bad_cases.append({
                    "stem": stem, "gts": gts, "preds": preds,
                    "error_type": "miss",
                    "detail": f"gt={gt['cls']}",
                    "iou": 0.0, "score": 0.5,
                })

        # 误检（预测无匹配）
        for pi, pred in enumerate(preds):
            if pi not in matched_pred:
                bad_cases.append({
                    "stem": stem, "gts": gts, "preds": preds,
                    "error_type": "false_positive",
                    "detail": f"pred={pred['cls']} conf={pred['conf']:.3f}",
                    "iou": 0.0, "score": 0.2,
                })

        # 低 IoU 匹配（0.3-0.5）
        for pi, pred in enumerate(preds):
            for gi, gt in enumerate(gts):
                if gi in matched_gt or pi in matched_pred:
                    continue
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if 0.3 <= iou < 0.5 and pred["cls"] == gt["cls"]:
                    bad_cases.append({
                        "stem": stem, "gts": gts, "preds": preds,
                        "error_type": "low_iou",
                        "detail": f"{gt['cls']} iou={iou:.3f}",
                        "iou": iou, "score": max(0, 0.5 - iou),
                    })

    # 优先挑 crazing / rolled-in_scale
    priority = {"crazing": 3, "rolled-in_scale": 2}
    for bc in bad_cases:
        for gt in bc["gts"]:
            bc["score"] += priority.get(gt["cls"], 0) * 0.1

    bad_cases.sort(key=lambda x: (-x["score"], x.get("iou", 0)))
    return bad_cases


def read_image_safe(path):
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_comparison(stem, gts, preds, error_type, detail, output_path):
    """画原图 + GT（绿框）+ 预测（红框）+ 标签。"""
    img_path = None
    for split in ["train", "validation"]:
        for cls in CLASSES:
            p = DATA_DIR / split / "images" / cls / f"{stem}.jpg"
            if p.exists():
                img_path = p
                break

    if img_path is None:
        return

    img = read_image_safe(img_path)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray")

    # GT: 绿色实线
    for gt in gts:
        rect = patches.Rectangle(
            (gt["bbox"][0], gt["bbox"][1]),
            gt["bbox"][2] - gt["bbox"][0], gt["bbox"][3] - gt["bbox"][1],
            linewidth=2, edgecolor="#00FF00", facecolor="none", linestyle="-",
        )
        ax.add_patch(rect)
        ax.text(gt["bbox"][0], gt["bbox"][1] - 3, f"GT:{gt['cls']}",
                fontsize=8, color="#00FF00", weight="bold")

    # Pred: 红色虚线
    for pred in preds:
        rect = patches.Rectangle(
            (pred["bbox"][0], pred["bbox"][1]),
            pred["bbox"][2] - pred["bbox"][0], pred["bbox"][3] - pred["bbox"][1],
            linewidth=2, edgecolor="#FF0000", facecolor="none", linestyle="--",
        )
        ax.add_patch(rect)
        ax.text(pred["bbox"][0], pred["bbox"][3] + 8,
                f"{pred['cls']} {pred['conf']:.2f}",
                fontsize=7, color="#FF0000")

    ax.set_title(f"{error_type}: {detail}", fontsize=10, color="#CC0000")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("Loading test data & running predictions...")
    gt_dict, preds_dict, img_paths = load_all_data()

    print("Analyzing errors...")
    bad_cases = analyze_errors(gt_dict, preds_dict)
    print(f"  Found {len(bad_cases)} potential bad cases")

    # 去重（同一 stem 只取最难的 1 个）
    seen_stems = set()
    selected = []
    error_counts = {"miss": 0, "false_positive": 0, "class_error": 0, "low_iou": 0}
    for bc in bad_cases:
        if bc["stem"] in seen_stems:
            continue
        if error_counts[bc["error_type"]] < 3:  # 每种错误至少 2 个
            seen_stems.add(bc["stem"])
            error_counts[bc["error_type"]] += 1
            selected.append(bc)
        if len(selected) >= 10:
            break

    print(f"  Selected top {len(selected)} unique cases")
    print()
    print("=" * 60)
    print(f"{'#':<3} {'Class':<20} {'Error':<15} {'Detail':<40}")
    print("-" * 60)

    descriptions = []
    for i, bc in enumerate(selected):
        main_cls = bc["gts"][0]["cls"] if bc["gts"] else "unknown"
        print(f"{i+1:<3} {main_cls:<20} {bc['error_type']:<15} {bc['detail']:<40}")

        # 画图
        out_path = OUT_DIR / f"bad_case_{i+1:02d}_{bc['error_type']}_{main_cls}.png"
        draw_comparison(bc["stem"], bc["gts"], bc["preds"],
                       bc["error_type"], bc["detail"], out_path)

        # 失败原因猜测
        if bc["error_type"] == "miss":
            reason = "模型对该类低对比度区域无响应，conf 低于阈值"
        elif bc["error_type"] == "false_positive":
            reason = "纹理噪声或边界被误判为该类特征"
        elif bc["error_type"] == "class_error":
            reason = "两类视觉特征相近（如 scratched vs crazing），混淆"
        elif bc["error_type"] == "low_iou":
            reason = "预测框覆盖不全（边界模糊或缺陷形状不规则）"
        else:
            reason = "待分析"

        descriptions.append({
            "id": i + 1,
            "class": main_cls,
            "error_type": bc["error_type"],
            "iou": f"{bc.get('iou', 0):.3f}",
            "reason": reason,
            "image": out_path.name,
        })

    print("-" * 60)
    print(f"\nPlots saved to {OUT_DIR}/")
    print("\n=== DEVLOG Entry ===")
    for d in descriptions:
        print(f"  {d['id']:>2}. {d['class']:<20} {d['error_type']:<15} IoU={d['iou']:>6} | {d['reason']}")
    print(f"  Plots: results/bad_cases/bad_case_*.png")


if __name__ == "__main__":
    main()
