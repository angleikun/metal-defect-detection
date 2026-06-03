"""NEU-DET 数据集分层 split。

按类分层，80/10/10 划分，seed=42 保证可复现。
输出 data/splits/{train,val,test}.txt，每行一个图像 stem。
脚本幂等：重复运行结果完全一致。
"""

import sys
import hashlib
from pathlib import Path
from collections import defaultdict

# ── 路径 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "NEU-DET"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from utils.seed import set_seed

set_seed(42)

CLASSES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]


def collect_valid_stems(data_dir: Path) -> dict[str, list[str]]:
    """收集所有有标注的图像 stem，按类分组并排序。

    同时扫描 data_dir 下 train/ 和 validation/ 两个子集，
    只保留 annotations/ 中有对应 XML 的图像。
    """
    stems_by_class: dict[str, list[str]] = {cls: [] for cls in CLASSES}

    for split in ["train", "validation"]:
        ann_dir = data_dir / split / "annotations"
        for cls in CLASSES:
            cls_dir = data_dir / split / "images" / cls
            if not cls_dir.is_dir():
                continue
            for img_path in sorted(cls_dir.glob("*.jpg")):
                stem = img_path.stem
                if (ann_dir / f"{stem}.xml").exists():
                    stems_by_class[cls].append(stem)

    # 排序（关键！os.listdir / rglob 顺序不可靠，
    # sorted() 保证跨平台/跨 Python 版本一致）
    for cls in CLASSES:
        stems_by_class[cls] = sorted(stems_by_class[cls])

    return stems_by_class


def extract_class(stem: str) -> str:
    """从 stem 提取类名。

    >>> extract_class("rolled-in_scale_200")
    'rolled-in_scale'
    >>> extract_class("crazing_1")
    'crazing'
    """
    # 长类名优先匹配，避免 "rolled-in" 误匹配 "rolled-in_scale"
    for cls in sorted(CLASSES, key=len, reverse=True):
        if stem.startswith(cls):
            remainder = stem[len(cls):]
            # remainder 应为空、纯数字、或 _数字
            if remainder == "" or remainder.replace("_", "").isdigit():
                return cls
    raise ValueError(f"Cannot extract class from stem: {stem!r}")


def split_stems(stems: list[str], n: int):
    """将 stems 按 80/10/10 划分。

    使用 round() 取整以保证总和等于 n；
    train 优先级最高，test 用减法兜底。
    """
    n_train = round(n * 0.8)
    n_val = round(n * 0.1)
    n_test = n - n_train - n_val
    return stems[:n_train], stems[n_train:n_train + n_val], stems[n_train + n_val:]


def compute_sha256(filepath: Path) -> str:
    return hashlib.sha256(filepath.read_bytes()).hexdigest()


# ── 主流程 ────────────────────────────────────────────
def main():
    # 1. 收集
    stems_by_class = collect_valid_stems(DATA_DIR)

    print("Valid images per class:")
    for cls in CLASSES:
        print(f"  {cls:<20} {len(stems_by_class[cls]):>4}")
    total = sum(len(v) for v in stems_by_class.values())
    print(f"  {'TOTAL':<20} {total:>4}")
    print()

    # 2. 分层 split
    train, val, test = [], [], []
    for cls in CLASSES:
        stems = stems_by_class[cls]
        t, v, te = split_stems(stems, len(stems))
        train.extend(t)
        val.extend(v)
        test.extend(te)

    # 3. 写出文件（内部按 stem 排序，便于 diff）
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    splits = {"train": train, "val": val, "test": test}
    for name, stems in splits.items():
        path = SPLITS_DIR / f"{name}.txt"
        path.write_text("\n".join(sorted(stems)) + "\n", encoding="utf-8")
        print(f"{name}.txt: {len(stems)} lines  ->  {path}")

    # 4. 验证：打印每个 split 的类分布
    print()
    for name in ["train", "val", "test"]:
        path = SPLITS_DIR / f"{name}.txt"
        stems_in_file = path.read_text(encoding="utf-8").splitlines()
        dist = defaultdict(int)
        for s in stems_in_file:
            dist[extract_class(s)] += 1
        print(f"{name} class distribution:")
        for cls in CLASSES:
            print(f"  {cls:<20} {dist[cls]:>4}")
        print(f"  {'TOTAL':<20} {len(stems_in_file):>4}")
        print()

    # 5. sha256 锁仓
    sha = compute_sha256(SPLITS_DIR / "train.txt")
    print(f"train.txt sha256: {sha}")
    print()

    # 凑 DEVLOG 用的一行
    class_totals = {cls: len(stems_by_class[cls]) for cls in CLASSES}
    print(
        "DEVLOG: Day 2.5 完成 stratified split，"
        f"train={len(train)} val={len(val)} test={len(test)}，"
        f"train.txt sha256 = {sha}"
    )


if __name__ == "__main__":
    main()
