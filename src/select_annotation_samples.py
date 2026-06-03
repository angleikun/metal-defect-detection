"""从 test set 每类随机抽 5 张（共 30 张），复制原图到 annotations_manual/images/。

用途：Day 5.5 人工标注 — U-Net 评测基准的前置步骤。
"""

import sys
from pathlib import Path
import shutil
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from utils.seed import set_seed

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "NEU-DET"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
OUT_IMAGES = PROJECT_ROOT / "data" / "annotations_manual" / "images"
OUT_SAMPLES_TXT = PROJECT_ROOT / "data" / "annotations_manual" / "samples.txt"

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def main():
    set_seed(42)
    rng = np.random.RandomState(42)

    # 读取 test set stems
    test_stems = set((SPLITS_DIR / "test.txt").read_text().splitlines())

    # 按类分组
    stems_by_class = {cls: [] for cls in CLASSES}
    for stem in test_stems:
        for cls in CLASSES:
            if stem.startswith(cls):
                stems_by_class[cls].append(stem)
                break

    # Build stem → image path
    stem_to_path = {}
    for split in ["train", "validation"]:
        for cls in CLASSES:
            cls_dir = DATA_DIR / split / "images" / cls
            if not cls_dir.is_dir():
                continue
            for img_path in cls_dir.glob("*.jpg"):
                stem_to_path[img_path.stem] = img_path

    # 创建输出目录
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)

    selected = []
    for cls in CLASSES:
        stems = sorted(stems_by_class[cls])
        n = min(5, len(stems))
        picks = rng.choice(stems, size=n, replace=False)
        for stem in picks:
            src = stem_to_path.get(stem)
            if src is None:
                print(f"  WARNING: {stem} not found in source images")
                continue
            dst = OUT_IMAGES / f"{stem}.jpg"
            if dst.exists():
                print(f"  SKIP (exists): {stem}.jpg")
            else:
                shutil.copy2(str(src), str(dst))
                print(f"  COPY: {stem}.jpg")
            selected.append(stem)

    # 写 samples.txt
    OUT_SAMPLES_TXT.write_text("\n".join(sorted(selected)) + "\n", encoding="utf-8")

    print()
    print(f"Total selected: {len(selected)} images")
    print(f"Images copied to: {OUT_IMAGES}")
    print(f"Sample list:      {OUT_SAMPLES_TXT}")
    print()
    print("Next: labelme data/annotations_manual/images --output data/annotations_manual/jsons --autosave --nodata")


if __name__ == "__main__":
    main()
