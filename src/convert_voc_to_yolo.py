"""NEU-DET 标注格式转换：PASCAL VOC XML → YOLO TXT。

输入：data/raw/NEU-DET/{train,validation}/annotations/*.xml
      data/splits/{train,val}.txt（图像 stem 列表）
输出：data/raw/NEU-DET/{train,validation}/labels/{class}/{stem}.txt（YOLO 归一化格式）
      data/splits/train_paths.txt / val_paths.txt（图像绝对路径列表，供 data.yaml 引用）
"""

from pathlib import Path
import xml.etree.ElementTree as ET

# ── 路径 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "NEU-DET"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

CLASSES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]
CLASS2ID = {cls: i for i, cls in enumerate(CLASSES)}


def build_stem_index(data_dir: Path) -> dict[str, dict]:
    """构建 stem → {image_path, xml_path} 索引。

    扫描 train/ 和 validation/ 两个子集，为每个有标注的图像
    记录其图像路径和 XML 路径。
    """
    index: dict[str, dict] = {}

    for split in ["train", "validation"]:
        ann_dir = data_dir / split / "annotations"
        for cls in CLASSES:
            cls_img_dir = data_dir / split / "images" / cls
            if not cls_img_dir.is_dir():
                continue
            for img_path in cls_img_dir.glob("*.jpg"):
                stem = img_path.stem
                xml_path = ann_dir / f"{stem}.xml"
                if xml_path.exists():
                    index[stem] = {
                        "image_path": img_path,
                        "xml_path": xml_path,
                        "split": split,
                        "cls": cls,
                    }
    return index


def convert_xml_to_yolo(xml_path: Path, img_w: int, img_h: int) -> list[str]:
    """将单个 PASCAL VOC XML 转为 YOLO 归一化行列表。

    每行格式：class_id x_center y_center width height
    所有坐标归一化到 [0, 1]。
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []

    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLASS2ID:
            print(f"  WARNING: unknown class {name!r} in {xml_path.name}, skipping")
            continue

        cls_id = CLASS2ID[name]
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)

        # 归一化
        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        # 裁剪到 [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))

        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return lines


def main():
    # 1. 构建 stem 索引
    print("Building stem index...")
    stem_index = build_stem_index(DATA_DIR)
    print(f"  Found {len(stem_index)} annotated images")

    # 2. 读取 split 文件
    train_stems = (SPLITS_DIR / "train.txt").read_text(encoding="utf-8").splitlines()
    val_stems = (SPLITS_DIR / "val.txt").read_text(encoding="utf-8").splitlines()
    print(f"  train: {len(train_stems)} stems")
    print(f"  val:   {len(val_stems)} stems")

    # 3. 转换
    for split_name, stems in [("train", train_stems), ("val", val_stems)]:
        converted = 0
        skipped = 0
        for stem in stems:
            if stem not in stem_index:
                print(f"  WARNING: {stem} not in index, skipping")
                skipped += 1
                continue

            info = stem_index[stem]
            xml_path = info["xml_path"]
            img_w, img_h = 200, 200  # NEU-DET 固定尺寸

            yolo_lines = convert_xml_to_yolo(xml_path, img_w, img_h)
            if not yolo_lines:
                print(f"  WARNING: no valid objects in {xml_path.name}")
                skipped += 1
                continue

            # 输出路径：labels/ 并行于 images/
            # image: train/images/crazing/crazing_1.jpg
            # label: train/labels/crazing/crazing_1.txt
            label_dir = DATA_DIR / info["split"] / "labels" / info["cls"]
            label_dir.mkdir(parents=True, exist_ok=True)
            label_path = label_dir / f"{stem}.txt"
            label_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
            converted += 1

        print(f"  {split_name}: converted {converted}, skipped {skipped}")

    # 4. 生成图像路径列表（供 data.yaml）
    for split_name, stems in [("train", train_stems), ("val", val_stems)]:
        path_list = []
        for stem in stems:
            if stem in stem_index:
                path_list.append(str(stem_index[stem]["image_path"].resolve()))
        out_path = SPLITS_DIR / f"{split_name}_paths.txt"
        out_path.write_text("\n".join(path_list) + "\n", encoding="utf-8")
        print(f"  {split_name}_paths.txt: {len(path_list)} lines")

    print()
    print("Done. Labels written, path lists generated.")
    print("Next: verify with -- python -c \"from ultralytics import YOLO; "
          "YOLO('yolov8n.pt')\" then run train.py")


if __name__ == "__main__":
    main()
