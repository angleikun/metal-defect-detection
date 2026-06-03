"""YOLOv8 训练脚本 — 金属表面缺陷检测。

Day 4 调参实验。Exp 4: 针对性提升 crazing 召回。
"""

import sys
import os
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.seed import set_seed
from ultralytics import YOLO

# ── Exp 4: 针对性提升 crazing ──────────────────────────
DATA_YAML = str(PROJECT_ROOT / "configs" / "data.yaml")
MODEL = "yolov8n.pt"
EPOCHS = 50
BATCH = 16
IMGSZ = 640
DEVICE = 0
WORKERS = 0
SEED = 42

set_seed(SEED)


def main():
    exp = "exp4_reproducibility"
    print("=" * 60)
    print(f"Reproducibility check: re-run Exp 4 with same seed=42")
    print("=" * 60)

    model = YOLO(MODEL)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        seed=SEED,
        deterministic=True,
        # ── Exp 4 关键修改 ──
        mosaic=0.0,            # 关！mosaic 切碎 crazing 网状纹理
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.0,             # 灰度图无需色相
        hsv_s=0.0,             # 饱和度也无意义
        hsv_v=0.3,             # 保留亮度抖动
        degrees=15,
        translate=0.1,
        scale=0.3,             # 缩小 scale 抖动（默认 0.5 对纹理太激进）
        fliplr=0.5,
        flipud=0.5,            # 钢板缺陷无方向性
        erasing=0.0,           # 关 random erasing，对纹理有害
        lr0=0.005,             # 小数据集默认 0.01 偏大
        patience=20,
        # ── 优化器 ──
        optimizer="auto",
        cos_lr=True,           # 复用 Exp 2 的 cosine
        close_mosaic=0,        # mosaic 已关，无需此参数
        # ── 保存 ──
        save=True,
        save_period=10,
        # ── 其他 ──
        project=str(PROJECT_ROOT / "models"),
        name=exp,
        exist_ok=True,
        pretrained=True,
        verbose=True,
        val=True,
        plots=False,
    )

    print()
    print(f"Best mAP@0.5:       {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"Best mAP@0.5:0.95:  {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"Model: models/{exp}/weights/best.pt")


if __name__ == "__main__":
    main()
