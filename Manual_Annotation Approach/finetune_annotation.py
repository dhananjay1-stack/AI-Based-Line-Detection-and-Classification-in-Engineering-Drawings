
import sys
import json
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    print("Install albumentations: pip install albumentations")
    sys.exit(1)

from model import DeepLabV3PlusEdge
from losses import CombinedLoss

# ─── Setup ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("finetune_annotation")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class AnnotationPatchDataset(Dataset):
    

    def __init__(self, root_dir, split="train", transform=None, num_classes=11):
        self.root_dir = Path(root_dir) / split
        self.img_dir = self.root_dir / "images"
        self.mask_dir = self.root_dir / "masks"
        self.transform = transform
        self.num_classes = num_classes

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        self.filenames = sorted([f.name for f in self.img_dir.glob("*.png")])
        logger.info(f"[{split}] Loaded {len(self.filenames)} patches from {self.root_dir}")

        # Compute class distribution for weighted sampling
        self._compute_class_stats()

    def _compute_class_stats(self):
        """Compute per-patch foreground stats for weighted sampling."""
        self.fg_ratios = []
        self.class_counts = defaultdict(int)

        for fname in self.filenames:
            mask_path = self.mask_dir / fname
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                fg_ratio = (mask > 0).sum() / mask.size
                self.fg_ratios.append(max(fg_ratio, 0.001))
                for cls_id in np.unique(mask):
                    if cls_id > 0:
                        self.class_counts[int(cls_id)] += 1
            else:
                self.fg_ratios.append(0.001)

    def get_sample_weights(self):
        """Return per-sample weights for WeightedRandomSampler."""
        # Weight by foreground ratio — patches with more foreground get higher weight
        weights = np.array(self.fg_ratios)
        # Boost: sqrt to moderate the weighting
        weights = np.sqrt(weights)
        # Normalize
        weights = weights / weights.sum() * len(weights)
        return weights

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # Load image (BGR → RGB)
        img = cv2.imread(str(self.img_dir / fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load mask
        mask = cv2.imread(str(self.mask_dir / fname), cv2.IMREAD_GRAYSCALE)

        # Clamp mask values to valid range
        mask = np.clip(mask, 0, self.num_classes - 1)

        # Generate edge target (1px skeleton of all foreground)
        fg_binary = (mask > 0).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(fg_binary, kernel, iterations=1)
        edge_target = fg_binary - eroded  # 1px boundary

        # Apply augmentation
        if self.transform:
            transformed = self.transform(image=img, masks=[mask, edge_target])
            img = transformed["image"]
            mask = transformed["masks"][0]
            edge_target = transformed["masks"][1]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask)
            edge_target = torch.from_numpy(edge_target)

        mask = mask.long()
        edge_target = edge_target.float().unsqueeze(0)  # (1, H, W)

        return img, mask, edge_target


# ─── Augmentation ─────────────────────────────────────────────────────────────

def get_train_transform(patch_size=512):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0,
                 mask_value=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussNoise(p=0.1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transform():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ─── Class Weights ────────────────────────────────────────────────────────────

def compute_class_weights(dataset, num_classes, bg_weight=0.3):
    """
    Compute inverse-frequency class weights from the dataset.
    Background (class 0) is explicitly down-weighted.
    """
    logger.info("Computing class weights from dataset...")
    pixel_counts = np.zeros(num_classes, dtype=np.float64)

    for i in tqdm(range(len(dataset)), desc="Scanning"):
        mask_path = dataset.mask_dir / dataset.filenames[i]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = np.clip(mask, 0, num_classes - 1)
        for c in range(num_classes):
            pixel_counts[c] += (mask == c).sum()

    # Avoid division by zero
    pixel_counts = np.maximum(pixel_counts, 1.0)
    total = pixel_counts.sum()

    # Inverse frequency
    weights = total / (num_classes * pixel_counts)

    # Cap extreme weights
    weights = np.clip(weights, 0.1, 50.0)

    # Down-weight background
    weights[0] = bg_weight

    # Normalize so mean = 1
    weights = weights / weights.mean()

    logger.info("Class weights:")
    for c in range(num_classes):
        pct = pixel_counts[c] / total * 100
        logger.info(f"  Class {c}: {pct:.2f}% of pixels, weight={weights[c]:.3f}")

    return torch.tensor(weights, dtype=torch.float32)


# ─── Validation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, val_loader, num_classes, device):
    """Run validation and compute per-class IoU."""
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for images, masks, _ in tqdm(val_loader, desc="[Validation]", leave=False):
        images = images.to(device)
        masks = masks.numpy()

        seg_logits, _ = model(images)
        preds = seg_logits.argmax(dim=1).cpu().numpy()

        for pred, gt in zip(preds, masks):
            valid = gt < num_classes
            pred_valid = pred[valid]
            gt_valid = gt[valid]
            np.add.at(confusion, (gt_valid, pred_valid), 1)

    # Compute per-class IoU
    ious = {}
    for c in range(num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        iou = tp / (tp + fp + fn + 1e-10)
        ious[c] = iou

    mean_iou = np.mean(list(ious.values()))
    return ious, mean_iou


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, grad_clip=1.0):
    """Train for one epoch with mixed precision."""
    model.train()
    running_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=True)
    for images, masks, edge_targets in pbar:
        images = images.to(device)
        masks = masks.to(device)
        edge_targets = edge_targets.to(device)

        optimizer.zero_grad()

        with autocast(enabled=(device == 'cuda')):
            seg_logits, edge_logits = model(images)
            loss, loss_dict = criterion(seg_logits, masks, edge_logits, edge_targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        n_batches += 1

        pbar.set_postfix({
            "loss": f"{running_loss / n_batches:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })

    return running_loss / max(n_batches, 1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune with manual annotations")

    # Paths
    parser.add_argument("--patch_dir", type=str,
                       default=r"C:\Users\Admin\line_detection\Approach 2\annotation_patches",
                       help="Directory with prepared patches")
    parser.add_argument("--checkpoint", type=str,
                       default=r"C:\Users\Admin\line_detection\Approach 2\finetune_output\checkpoints\best_model.pth",
                       help="Checkpoint to load")
    parser.add_argument("--out_dir", type=str,
                       default=r"C:\Users\Admin\line_detection\Approach 2\finetune_annotation_output",
                       help="Output directory")

    # Model
    parser.add_argument("--num_classes", type=int, default=11,
                       help="Number of classes (10 original + Section_hatching = 11)")
    parser.add_argument("--old_num_classes", type=int, default=10,
                       help="Number of classes in the checkpoint being loaded")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--no_edge_head", action="store_true",
                       help="Disable edge/centerline auxiliary head")

    # Training phases
    parser.add_argument("--phase1_epochs", type=int, default=10,
                       help="Phase 1: frozen encoder epochs")
    parser.add_argument("--phase2_epochs", type=int, default=40,
                       help="Phase 2: full fine-tuning epochs")
    parser.add_argument("--phase1_lr", type=float, default=3e-4)
    parser.add_argument("--phase2_lr_backbone", type=float, default=1e-5)
    parser.add_argument("--phase2_lr_decoder", type=float, default=5e-5)
    parser.add_argument("--phase2_lr_heads", type=float, default=1e-4)

    # Training options
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bg_weight", type=float, default=0.3,
                       help="Background class weight (lower = less emphasis on BG)")
    parser.add_argument("--early_stopping", type=int, default=15,
                       help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                       help="Resume from last checkpoint")

    args = parser.parse_args()

    seed_everything(args.seed)

    # ── Setup output ──
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    preview_dir = out_dir / "previews"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    # File logging
    fh = logging.FileHandler(out_dir / "training.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    logger.info("=" * 60)
    logger.info("FINE-TUNING WITH MANUAL ANNOTATIONS")
    logger.info("=" * 60)
    logger.info(f"Patch dir:     {args.patch_dir}")
    logger.info(f"Checkpoint:    {args.checkpoint}")
    logger.info(f"Output:        {args.out_dir}")
    logger.info(f"Num classes:   {args.num_classes}")
    logger.info(f"Device:        {DEVICE}")

    # ── Load class info ──
    classes_path = Path(args.patch_dir) / "classes.json"
    if classes_path.exists():
        with open(classes_path) as f:
            class_mapping = json.load(f)
        logger.info(f"Classes: {class_mapping}")
    else:
        logger.warning("No classes.json found, using default mapping")
        class_mapping = {"Background": 0}

    # Also load master classes for ID→name mapping
    master_path = Path(args.patch_dir) / "master_classes.json"
    id_to_name = {0: "Background"}
    if master_path.exists():
        with open(master_path) as f:
            master = json.load(f)
        for name, cid in master.items():
            id_to_name[cid] = name
    else:
        # Fallback
        default_names = {
            0: "Background", 1: "Center_line", 2: "Dimension_lines",
            3: "Extension_line", 4: "Feature_Visible", 5: "Leader_line",
            6: "Phantom_lines", 7: "break_line", 8: "cutting_plane",
            9: "hidden_lines", 10: "Section_hatching",
        }
        id_to_name = default_names

    # ── Create datasets ──
    use_edge = not args.no_edge_head

    train_dataset = AnnotationPatchDataset(
        args.patch_dir, split="train",
        transform=get_train_transform(),
        num_classes=args.num_classes,
    )
    val_dataset = AnnotationPatchDataset(
        args.patch_dir, split="val",
        transform=get_val_transform(),
        num_classes=args.num_classes,
    )

    # Weighted sampler
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=sampler, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )

    logger.info(f"Train: {len(train_dataset)} patches, Val: {len(val_dataset)} patches")

    # ── Compute class weights ──
    class_weights = compute_class_weights(train_dataset, args.num_classes, args.bg_weight)
    class_weights = class_weights.to(DEVICE)

    # ── Build model ──
    model = DeepLabV3PlusEdge(
        num_classes=args.num_classes,
        backbone=args.backbone,
        use_edge_head=use_edge,
    )

    # Load checkpoint
    if os.path.exists(args.checkpoint):
        model.load_pretrained_checkpoint(
            args.checkpoint,
            old_num_classes=args.old_num_classes,
            device=DEVICE,
        )
    else:
        logger.warning(f"Checkpoint not found: {args.checkpoint}")
        logger.info("Training from ImageNet pretrained weights only")

    model = model.to(DEVICE)
    model.summary()

    # ── Loss ──
    criterion = CombinedLoss(
        num_classes=args.num_classes,
        class_weights=class_weights,
        ce_weight=1.0,
        dice_weight=1.0,
        focal_weight=0.5,
        edge_weight=0.5 if use_edge else 0.0,
        focal_gamma=2.0,
        use_edge=use_edge,
    )

    scaler = GradScaler()

    # ── Training history ──
    best_miou = 0.0
    epochs_no_improve = 0
    history = []

    # Resume check
    start_phase = 1
    start_epoch = 0
    if args.resume:
        last_ckpt = ckpt_dir / "last_model.pth"
        if last_ckpt.exists():
            ckpt = torch.load(str(last_ckpt), map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            best_miou = ckpt.get("best_miou", 0.0)
            start_phase = 2 if ckpt.get("phase", "phase1") == "phase2" else 1
            start_epoch = ckpt.get("epoch", 0) + 1
            logger.info(f"Resumed from epoch {start_epoch}, phase {start_phase}, mIoU {best_miou:.4f}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: Frozen Encoder
    # ══════════════════════════════════════════════════════════════════

    if start_phase <= 1 and args.phase1_epochs > 0:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: Frozen Encoder Training")
        logger.info("=" * 60)

        model.freeze_encoder()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.phase1_lr,
            weight_decay=1e-4,
        )

        # Cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.phase1_epochs, eta_min=1e-6,
        )

        for epoch in range(start_epoch if start_phase == 1 else 0, args.phase1_epochs):
            avg_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler,
                DEVICE, epoch + 1, grad_clip=1.0,
            )

            # Validate
            ious, miou = validate(model, val_loader, args.num_classes, DEVICE)

            scheduler.step()

            # Log
            logger.info(f"  {'Class':<30s} {'IoU':>8s}")
            logger.info(f"  {'-' * 38}")
            for c in range(args.num_classes):
                name = id_to_name.get(c, f"class_{c}")
                if ious[c] > 0 or c == 0:
                    logger.info(f"  {name:<30s} {ious[c]:.4f}")
            logger.info(f"  {'-' * 38}")
            logger.info(f"  {'MEAN IoU':<30s} {miou:.4f}")

            history.append({
                "epoch": epoch + 1, "phase": "phase1",
                "loss": avg_loss, "miou": miou,
            })

            # Save best
            if miou > best_miou:
                best_miou = miou
                epochs_no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "phase": "phase1",
                    "model_state_dict": model.state_dict(),
                    "best_miou": best_miou,
                    "num_classes": args.num_classes,
                    "class_mapping": class_mapping,
                }, str(ckpt_dir / "best_model.pth"))
                logger.info(f"  ★ New best mIoU: {best_miou:.4f}")

            # Save last
            torch.save({
                "epoch": epoch,
                "phase": "phase1",
                "model_state_dict": model.state_dict(),
                "best_miou": best_miou,
                "num_classes": args.num_classes,
            }, str(ckpt_dir / "last_model.pth"))

        start_epoch = 0  # Reset for phase 2

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: Full Fine-Tuning with Differential LR
    # ══════════════════════════════════════════════════════════════════

    if args.phase2_epochs > 0:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: Full Fine-Tuning")
        logger.info("=" * 60)

        model.unfreeze_all()

        param_groups = model.get_param_groups(
            lr_backbone=args.phase2_lr_backbone,
            lr_decoder=args.phase2_lr_decoder,
            lr_heads=args.phase2_lr_heads,
        )

        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.phase2_epochs, eta_min=1e-7,
        )

        # Warmup for first 3 epochs
        warmup_epochs = 3

        for epoch in range(start_epoch if start_phase == 2 else 0, args.phase2_epochs):
            # Manual warmup
            if epoch < warmup_epochs:
                warmup_factor = (epoch + 1) / warmup_epochs
                for pg in optimizer.param_groups:
                    if pg.get("name") == "backbone":
                        pg["lr"] = args.phase2_lr_backbone * warmup_factor
                    elif pg.get("name") == "decoder":
                        pg["lr"] = args.phase2_lr_decoder * warmup_factor
                    else:
                        pg["lr"] = args.phase2_lr_heads * warmup_factor

            avg_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler,
                DEVICE, epoch + 1, grad_clip=1.0,
            )

            # Validate
            ious, miou = validate(model, val_loader, args.num_classes, DEVICE)

            if epoch >= warmup_epochs:
                scheduler.step()

            # Log
            logger.info(f"  {'Class':<30s} {'IoU':>8s}")
            logger.info(f"  {'-' * 38}")
            for c in range(args.num_classes):
                name = id_to_name.get(c, f"class_{c}")
                if ious[c] > 0 or c == 0:
                    logger.info(f"  {name:<30s} {ious[c]:.4f}")
            logger.info(f"  {'-' * 38}")
            logger.info(f"  {'MEAN IoU':<30s} {miou:.4f}")

            history.append({
                "epoch": epoch + 1, "phase": "phase2",
                "loss": avg_loss, "miou": miou,
            })

            # Save best
            if miou > best_miou:
                best_miou = miou
                epochs_no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "phase": "phase2",
                    "model_state_dict": model.state_dict(),
                    "best_miou": best_miou,
                    "num_classes": args.num_classes,
                    "class_mapping": class_mapping,
                }, str(ckpt_dir / "best_model.pth"))
                logger.info(f"  ★ New best mIoU: {best_miou:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.early_stopping:
                    logger.info(f"  Early stopping at epoch {epoch + 1}")
                    break

            # Save last
            torch.save({
                "epoch": epoch,
                "phase": "phase2",
                "model_state_dict": model.state_dict(),
                "best_miou": best_miou,
                "num_classes": args.num_classes,
            }, str(ckpt_dir / "last_model.pth"))

    # ── Save training history ──
    history_path = out_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # ── Final Summary ──
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Best mIoU:    {best_miou:.4f}")
    logger.info(f"  Best model:   {ckpt_dir / 'best_model.pth'}")
    logger.info(f"  History:      {history_path}")
    logger.info(f"\nTo run inference:")
    logger.info(f"  python inference.py --checkpoint \"{ckpt_dir / 'best_model.pth'}\" "
                f"--input <drawing.png> --num_classes {args.num_classes}")


if __name__ == "__main__":
    main()
