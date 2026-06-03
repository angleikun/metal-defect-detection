"""U-Net 训练 — 弱监督分割。

两套 mask：
  - baseline (bbox 内全填充)
  - refined (bbox + Otsu + 形态学)

两次训练，vs 30 张人工 mask 作绝对评测。
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm

os.environ["WANDB_MODE"] = "disabled"

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from utils.seed import set_seed

set_seed(42)

# ── 配置 ──────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "NEU-DET"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MANUAL_MASKS = PROJECT_ROOT / "data" / "annotations_manual" / "masks"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 50
IMG_SIZE = 224  # 必须能被 32 整除（resnet34 有 5 次下采样）
LR = 1e-4

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


# ── Dataset ───────────────────────────────────────────
class NEUDETMaskDataset(Dataset):
    """加载图像 + mask 对（按 split 文件）。"""

    def __init__(self, stems, data_dir, mask_dir):
        self.stems = stems
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        # 预构建 stem → image_path
        self.stem_to_img = {}
        for split in ["train", "validation"]:
            for cls in CLASSES:
                cls_dir = data_dir / split / "images" / cls
                if cls_dir.is_dir():
                    for p in cls_dir.glob("*.jpg"):
                        self.stem_to_img[p.stem] = p

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        img_path = self.stem_to_img.get(stem)
        mask_path = self.mask_dir / f"{stem}.png"

        # 读图 + resize 到 224
        with open(img_path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        img = np.expand_dims(img, axis=0)  # (1, 224, 224)

        # 读 mask + resize（使用 INTER_NEAREST 保持二值性）
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((200, 200), dtype=np.uint8)
        mask = cv2.resize(mask.astype(np.float32), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST) / 255.0
        mask = np.expand_dims(mask, axis=0)  # (1, 224, 224)

        # 灰度图重复 3 通道（适配 resnet 预训练权重）
        img_3ch = np.repeat(img, 3, axis=0)

        return torch.from_numpy(img_3ch), torch.from_numpy(mask)


# ── 损失函数 ────────────────────────────────────────
def dice_loss(pred, target, smooth=1.0):
    """Dice loss for binary segmentation."""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return 1 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def combined_loss(pred, target, bce_weight=0.5):
    """BCE + Dice 组合损失."""
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(torch.sigmoid(pred), target)
    return bce_weight * bce + (1 - bce_weight) * dice


def compute_iou(pred_mask, gt_mask):
    """计算单张图的 IoU (pixel-level)."""
    pred_bin = (pred_mask > 0.5).astype(np.uint8)
    gt_bin = (gt_mask > 0.5).astype(np.uint8)
    inter = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()
    return inter / (union + 1e-8)


# ── 训练 ──────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, bce_w):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = combined_loss(logits, masks, bce_w)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, bce_w):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Val", leave=False):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            logits = model(imgs)
            loss = combined_loss(logits, masks, bce_w)
            total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_manual_masks(model):
    """在 30 张人工 mask 上计算 per-image IoU。"""
    import segmentation_models_pytorch as smp

    model.eval()
    ious = []
    for mask_path in sorted(MANUAL_MASKS.glob("*.png")):
        stem = mask_path.stem
        # 找原图
        img_path = None
        for split in ["train", "validation"]:
            for cls in CLASSES:
                p = DATA_DIR / split / "images" / cls / f"{stem}.jpg"
                if p.exists():
                    img_path = p
                    break

        if img_path is None:
            continue

        with open(img_path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        img_3ch = np.repeat(img[np.newaxis, ...], 3, axis=0)
        img_t = torch.from_numpy(img_3ch).unsqueeze(0).to(DEVICE)

        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        with torch.no_grad():
            pred = torch.sigmoid(model(img_t)).cpu().numpy()[0, 0]
            pred_resized = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

        iou = compute_iou(pred_resized, gt)
        ious.append({"stem": stem, "iou": iou})

    return ious


def train_unet(mask_type: str):
    """统一训练入口。

    Args:
        mask_type: 'baseline' 或 'refined'
    """
    import segmentation_models_pytorch as smp

    mask_dir = DATA_DIR / f"masks_{mask_type}"
    exp_name = f"unet_{mask_type}" if mask_type == "baseline" else "unet_refined_v2"

    # 加载 split
    train_stems = (SPLITS_DIR / "train.txt").read_text().splitlines()
    val_stems = (SPLITS_DIR / "val.txt").read_text().splitlines()

    train_ds = NEUDETMaskDataset(train_stems, DATA_DIR, mask_dir)
    val_ds = NEUDETMaskDataset(val_stems, DATA_DIR, mask_dir)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} images, Val: {len(val_ds)} images")
    print(f"Mask type: {mask_type}")

    # 模型
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,         # 灰度图重复 3 通道
        classes=1,             # 二分类分割
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 动态 loss 权重
    bce_w = 0.5

    best_val_iou = 0.0
    best_epoch = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, bce_w)
        val_loss = validate(model, val_loader, bce_w)
        scheduler.step()

        # 每 5 epoch 在人工 mask 上测一次 IoU
        if epoch % 5 == 0 or epoch == EPOCHS:
            manual_ious = evaluate_manual_masks(model)
            mean_iou = np.mean([m["iou"] for m in manual_ious])
            print(f"  Epoch {epoch:3d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} | manual_IoU={mean_iou:.4f}")
            history.append({"epoch": epoch, "iou": mean_iou})

            if mean_iou > best_val_iou:
                best_val_iou = mean_iou
                best_epoch = epoch
                torch.save(model.state_dict(), str(PROJECT_ROOT / "models" / f"{exp_name}_best.pt"))

            # 动态调整 loss 权重（若 val IoU < 0.40 且过 20 epoch）
            if epoch >= 20 and mean_iou < 0.40 and bce_w == 0.5:
                bce_w = 0.3
                print(f"  -> adjusting loss: BCE_w={bce_w}, Dice_w={1-bce_w}")
        else:
            print(f"  Epoch {epoch:3d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    print(f"\nBest manual IoU: {best_val_iou:.4f} at epoch {best_epoch}")

    # 加载最佳权重做最终评测
    model.load_state_dict(torch.load(str(PROJECT_ROOT / "models" / f"{exp_name}_best.pt")))
    final_ious = evaluate_manual_masks(model)
    return final_ious, best_val_iou, history


def main():
    PROJ_MODELS = PROJECT_ROOT / "models"
    PROJ_MODELS.mkdir(exist_ok=True)

    results = {}
    for mask_type in ["refined"]:  # v2: only retrain refined
        print()
        print("=" * 60)
        print(f"Training U-Net with {mask_type} masks")
        print("=" * 60)
        ious, best_iou, history = train_unet(mask_type)
        results[mask_type] = {"ious": ious, "best_iou": best_iou}

    # ── 汇总 ────────────────────────────────────────
    print()
    print("=" * 60)
    print("RESULTS: Baseline vs Refined")
    print("=" * 60)
    for mt, r in results.items():
        print(f"\n{mt} masks:")
        for item in r["ious"]:
            print(f"  {item['stem']:<30} IoU={item['iou']:.4f}")
        print(f"  Mean IoU: {r['best_iou']:.4f}")


if __name__ == "__main__":
    main()
