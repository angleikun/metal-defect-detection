"""弱监督伪 mask 生成器。

从 bbox + Otsu 自适应阈值 + 形态学 refine 生成像素级 mask。
同时生成 baseline mask（bbox 内全填充，作对照）。

用法：python src/mask_generator.py
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "NEU-DET"
OUT_BASELINE = PROJECT_ROOT / "data" / "raw" / "NEU-DET" / "masks_baseline"
OUT_REFINED = PROJECT_ROOT / "data" / "raw" / "NEU-DET" / "masks_refined"

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

# 形态学参数
KERNEL_OPEN = 3       # 开运算核大小（去噪）
KERNEL_CLOSE = 5      # 闭运算核大小（连通）
OTSU_BIAS = -0.10      # Otsu 阈值偏置（负值 = 降低阈值，保留更多前景）


def read_gray(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def otsu_refine_roi(roi_gray: np.ndarray, bias: float = OTSU_BIAS) -> np.ndarray:
    """对 ROI 灰度图做 Otsu + 形态学 refine，返回二值 mask。"""
    # Otsu 自适应阈值
    thr, _ = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(0, min(255, thr + int(bias * 255)))

    _, binary = cv2.threshold(roi_gray, thr, 255, cv2.THRESH_BINARY)
    binary_u8 = binary.astype(np.uint8)

    # 形态学开运算（去噪点）
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_OPEN, KERNEL_OPEN))
    opened = cv2.morphologyEx(binary_u8, cv2.MORPH_OPEN, kernel_open)

    # 形态学闭运算（连通断开的缺陷区域）
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_CLOSE, KERNEL_CLOSE))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

    return closed


def generate_masks(xml_path: Path, img_path: Path):
    """为单张图生成 baseline + refined 两套 mask。

    Returns: (baseline_mask, refined_mask) or (None, None) on failure.
    """
    # 读图
    img = read_gray(img_path)
    if img is None:
        return None, None
    h, w = img.shape

    tree = ET.parse(xml_path)
    root = tree.getroot()

    baseline = np.zeros((h, w), dtype=np.uint8)
    refined = np.zeros((h, w), dtype=np.uint8)

    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLASSES:
            continue

        bb = obj.find("bndbox")
        xmin = int(float(bb.find("xmin").text))
        ymin = int(float(bb.find("ymin").text))
        xmax = int(float(bb.find("xmax").text))
        ymax = int(float(bb.find("ymax").text))

        # 确保在图像内
        xmin = max(0, xmin); ymin = max(0, ymin)
        xmax = min(w, xmax); ymax = min(h, ymax)

        if xmax <= xmin or ymax <= ymin:
            continue

        # Baseline: bbox 内全填充
        baseline[ymin:ymax, xmin:xmax] = 255

        # Refined: ROI 内 Otsu + 形态学
        roi = img[ymin:ymax, xmin:xmax]
        roi_mask = otsu_refine_roi(roi)
        refined[ymin:ymax, xmin:xmax] = np.maximum(
            refined[ymin:ymax, xmin:xmax], roi_mask
        )

    return baseline, refined


def main():
    OUT_BASELINE.mkdir(parents=True, exist_ok=True)
    OUT_REFINED.mkdir(parents=True, exist_ok=True)

    done = 0
    for split in ["train", "validation"]:
        ann_dir = DATA_DIR / split / "annotations"
        for xml_path in ann_dir.glob("*.xml"):
            stem = xml_path.stem
            # 找对应图像
            img_path = None
            for cls in CLASSES:
                p = DATA_DIR / split / "images" / cls / f"{stem}.jpg"
                if p.exists():
                    img_path = p
                    break
            if img_path is None:
                continue

            baseline, refined = generate_masks(xml_path, img_path)
            if baseline is None:
                continue

            # 写入：用 stem 命名，保存到 masks_baseline/ 和 masks_refined/
            cv2.imwrite(str(OUT_BASELINE / f"{stem}.png"), baseline)
            cv2.imwrite(str(OUT_REFINED / f"{stem}.png"), refined)
            done += 1

            if done % 300 == 0:
                print(f"  {done} masks done...")

    print(f"\nGenerated {done} mask pairs")
    print(f"  Baseline: {OUT_BASELINE}")
    print(f"  Refined:  {OUT_REFINED}")


if __name__ == "__main__":
    main()
