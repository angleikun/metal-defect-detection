"""LabelMe JSON polygon → PNG mask。

输入：data/annotations_manual/jsons/*.json
输出：data/annotations_manual/masks/*.png（0=背景, 255=缺陷）
"""

from pathlib import Path
import json
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JSONS_DIR = PROJECT_ROOT / "data" / "annotations_manual" / "jsons"
MASKS_DIR = PROJECT_ROOT / "data" / "annotations_manual" / "masks"

MASKS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    done = 0
    for json_path in sorted(JSONS_DIR.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        h = data["imageHeight"]
        w = data["imageWidth"]
        mask = np.zeros((h, w), dtype=np.uint8)

        for shape in data["shapes"]:
            if shape["shape_type"] != "polygon":
                continue
            pts = np.array(shape["points"], dtype=np.int32)
            # fillPoly：多边形内部填 255
            cv2.fillPoly(mask, [pts], color=255)

        stem = json_path.stem
        out_path = MASKS_DIR / f"{stem}.png"
        cv2.imwrite(str(out_path), mask)
        fg_ratio = mask.sum() / (h * w * 255) * 100
        flag = " ⚠ low" if fg_ratio < 0.5 else ""
        print(f"  {stem}.png: {mask.sum()/255:.0f} fg px ({fg_ratio:.2f}%){flag}")
        done += 1

    print(f"\nGenerated {done} PNG masks in {MASKS_DIR}")


if __name__ == "__main__":
    main()
