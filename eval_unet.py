"""评测 baseline U-Net vs 30 张真人工 mask。"""

import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from utils.seed import set_seed

set_seed(42)

DEVICE = torch.device("cuda:0")
IMG_SIZE = 224
MASKS_DIR = PROJECT_ROOT / "data" / "annotations_manual" / "masks"
IMAGES_DIR = PROJECT_ROOT / "data" / "annotations_manual" / "images"
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

# Load model
model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1).to(DEVICE)
model.load_state_dict(torch.load(str(PROJECT_ROOT / "models" / "unet_baseline_best.pt"), map_location=DEVICE))
model.eval()

results = []
all_gt_fg = 0
all_pred_fg = 0
all_inter = 0
all_union = 0

for img_path in sorted(IMAGES_DIR.glob("*.jpg")):
    stem = img_path.stem
    mask_path = MASKS_DIR / f"{stem}.png"
    if not mask_path.exists():
        continue

    # GT mask (200x200)
    gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    gt_bin = (gt > 0).astype(np.uint8)

    # Image: grayscale → resize to 224 → repeat 3ch → normalize /255
    with open(img_path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img_3ch = np.repeat(img[np.newaxis, ...], 3, axis=0)
    img_t = torch.from_numpy(img_3ch).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        pred_224 = torch.sigmoid(model(img_t)).cpu().numpy()[0, 0]

    # Threshold → resize 200 with INTER_NEAREST
    pred_bin_224 = (pred_224 > 0.5).astype(np.uint8)
    pred_bin = cv2.resize(pred_bin_224, (200, 200), interpolation=cv2.INTER_NEAREST)

    # Per-image IoU
    inter = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()
    iou = inter / union if union > 0 else 1.0

    # Accumulate for micro
    all_gt_fg += int(gt_bin.sum())
    all_pred_fg += int(pred_bin.sum())
    all_inter += int(inter)
    all_union += int(union)

    cls = None
    for c in CLASSES:
        if stem.startswith(c):
            cls = c
            break

    results.append({
        "stem": stem, "class": cls,
        "gt_fg_ratio": gt_bin.sum() / gt_bin.size * 100,
        "pred_fg_ratio": pred_bin.sum() / pred_bin.size * 100,
        "iou": iou,
    })

# ── Table A ────────────────────────────────────────────
print("=== Table A: Per-Image IoU ===")
print(f"{'filename':<32} {'class':<20} {'gt_fg%':>8} {'pred_fg%':>8} {'IoU':>8}")
print("-" * 80)
for r in results:
    print(f"{r['stem']+'.png':<32} {r['class']:<20} {r['gt_fg_ratio']:>7.1f}% {r['pred_fg_ratio']:>7.1f}% {r['iou']:>8.4f}")

# Save CSV
csv_path = PROJECT_ROOT / "results" / "unet_baseline_human_eval.csv"
csv_path.parent.mkdir(exist_ok=True)
csv_path.write_text(
    "filename,class,gt_fg_ratio,pred_fg_ratio,iou\n"
    + "\n".join(f"{r['stem']}.png,{r['class']},{r['gt_fg_ratio']:.2f},{r['pred_fg_ratio']:.2f},{r['iou']:.4f}" for r in results)
)
print(f"\nSaved: {csv_path}")

# ── Table B ────────────────────────────────────────────
print()
print("=== Table B: Per-Class IoU ===")
print(f"{'class':<20} {'mean IoU':>8} {'std':>8} {'min':>8} {'max':>8}")
print("-" * 60)
per_class = {c: [r["iou"] for r in results if r["class"] == c] for c in CLASSES}
for cls in CLASSES:
    vals = per_class[cls]
    if vals:
        print(f"{cls:<20} {np.mean(vals):>8.4f} {np.std(vals):>8.4f} {np.min(vals):>8.4f} {np.max(vals):>8.4f}")

# ── Table C ────────────────────────────────────────────
micro_iou = all_inter / all_union if all_union > 0 else 1.0
macro_iou = np.mean([np.mean(per_class[c]) for c in CLASSES if per_class[c]])

print()
print("=== Table C: Summary ===")
print(f"Micro IoU (pixel-level global): {micro_iou:.4f}")
print(f"Macro IoU (per-class mean):     {macro_iou:.4f}")
print(f"Total GT fg pixels:             {all_gt_fg}")
print(f"Total pred fg pixels:           {all_pred_fg}")
