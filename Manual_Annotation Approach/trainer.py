"""
Fine-Tuning Trainer for Thin-Line Segmentation
===============================================

Two-phase training pipeline:
  Phase 1: Encoder frozen, train decoder + heads (fast adaptation)
  Phase 2: Full network with differential learning rates (refinement)

Usage:
    python trainer.py                           # Run with defaults from config.py
    python trainer.py --phase1_epochs 10         # Override specific settings
    python trainer.py --no_edge_head             # Disable edge/centerline head
"""

import os
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

from config import TrainConfig, ClassInfo
from model import DeepLabV3PlusEdge
from losses import CombinedLoss

# ─── Setup ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("trainer")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PatchDataset(Dataset):
    """
    Loads pre-extracted patches (images, masks, centerlines) from disk.
    
    Uses metadata CSV for split filtering and sampling weights.
    Images are JPEG, masks and centerlines are PNG with class IDs.
    """

    def __init__(self, patch_root, split="train", transform=None, load_edge=True):
        self.root = Path(patch_root)
        self.img_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.skel_dir = self.root / "centerlines"
        self.transform = transform
        self.load_edge = load_edge

        # Load metadata
        meta_path = self.root / "metadata" / "patch_metadata.csv"
        meta_df = pd.read_csv(meta_path)
        self.meta = meta_df[meta_df["split"] == split].reset_index(drop=True)

        logger.info(f"PatchDataset [{split}]: {len(self.meta)} patches")

    def __len__(self):
        return len(self.meta)

    def get_sampling_weights(self):
        """Return sampling weights for WeightedRandomSampler."""
        return self.meta["sampling_weight"].values.astype(np.float64)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        name = row["patch_name"]

        # Load image (JPEG)
        img_path = self.img_dir / f"{name}.jpg"
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (PNG, class IDs)
        mask_path = self.mask_dir / f"{name}.png"
        mask = np.array(Image.open(mask_path))

        # Load skeleton/centerline (PNG, class IDs → binarize)
        edge_mask = None
        if self.load_edge:
            skel_path = self.skel_dir / f"{name}.png"
            if skel_path.exists():
                skel = np.array(Image.open(skel_path))
                edge_mask = (skel > 0).astype(np.uint8)  # Binary
            else:
                edge_mask = np.zeros_like(mask, dtype=np.uint8)

        # Apply augmentations
        if self.transform:
            if edge_mask is not None:
                aug = self.transform(image=image, masks=[mask, edge_mask])
                image = aug["image"]
                mask = aug["masks"][0]
                edge_mask = aug["masks"][1]
            else:
                aug = self.transform(image=image, mask=mask)
                image = aug["image"]
                mask = aug["mask"]

        result = {
            "image": image,
            "mask": mask.long() if isinstance(mask, torch.Tensor) else torch.tensor(mask, dtype=torch.long),
        }

        if edge_mask is not None:
            if isinstance(edge_mask, torch.Tensor):
                result["edge"] = edge_mask.unsqueeze(0).float()
            else:
                result["edge"] = torch.tensor(edge_mask, dtype=torch.float32).unsqueeze(0)

        return result


# ─── Augmentations ────────────────────────────────────────────────────────────

def get_train_transform(cfg):
    transforms = []
    if cfg.aug_hflip:
        transforms.append(A.HorizontalFlip(p=0.5))
    if cfg.aug_vflip:
        transforms.append(A.VerticalFlip(p=0.5))
    if cfg.aug_rotate_limit > 0:
        transforms.append(A.Rotate(limit=cfg.aug_rotate_limit, p=0.5,
                                    border_mode=cv2.BORDER_CONSTANT))
    if cfg.aug_brightness_contrast:
        transforms.append(A.RandomBrightnessContrast(p=0.2))

    transforms.extend([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return A.Compose(transforms)


def get_val_transform():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ─── Class Weights ────────────────────────────────────────────────────────────

def compute_class_weights(dataset, num_classes, n_samples=500):
    """
    Compute inverse-frequency class weights from a subset of masks.
    Critical for thin-line segmentation where background dominates.
    """
    logger.info(f"Computing class weights from {min(n_samples, len(dataset))} samples...")
    counts = np.zeros(num_classes, dtype=np.float64)
    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    for i in indices:
        sample = dataset[i]
        mask = sample["mask"].numpy()
        unique, ucounts = np.unique(mask, return_counts=True)
        for u, c in zip(unique, ucounts):
            if u < num_classes:
                counts[u] += c

    total = counts.sum() + 1e-6
    freq = counts / total
    weights = 1.0 / (freq + 0.01)
    weights = weights / weights.max()  # Normalize to [0, 1]

    logger.info(f"  Class weights: {np.round(weights, 3)}")
    return torch.tensor(weights, dtype=torch.float32)


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_iou_per_class(pred, target, num_classes):
    """Compute IoU for each class."""
    ious = []
    for cls in range(num_classes):
        p = (pred == cls)
        g = (target == cls)
        intersection = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)
    return ious


# ─── Optimizer + Scheduler ────────────────────────────────────────────────────

def build_optimizer_phase1(model, cfg):
    """Phase 1: Only trainable params (decoder + heads), single LR."""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.phase1_lr, weight_decay=cfg.weight_decay)
    return optimizer


def build_optimizer_phase2(model, cfg):
    """Phase 2: Differential LR for backbone vs decoder vs heads."""
    param_groups = model.get_param_groups(
        lr_backbone=cfg.phase2_lr_backbone,
        lr_decoder=cfg.phase2_lr_decoder,
        lr_heads=cfg.phase2_lr_heads,
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    return optimizer


def build_scheduler(optimizer, total_epochs, warmup_epochs, steps_per_epoch, mode="cosine"):
    """
    Cosine annealing with linear warmup.
    """
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, scaler,
                    cfg, epoch, phase_name):
    """Train for one epoch with AMP and gradient accumulation."""
    model.train()
    running_loss = defaultdict(float)
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"[{phase_name}] Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(DEVICE)
        masks = batch["mask"].to(DEVICE)
        edges = batch.get("edge")
        if edges is not None:
            edges = edges.to(DEVICE)

        with autocast(enabled=cfg.use_amp):
            seg_logits, edge_logits = model(images)
            total_loss, loss_dict = criterion(
                seg_logits, masks,
                edge_logits=edge_logits,
                edge_targets=edges,
            )
            total_loss = total_loss / cfg.grad_accum_steps

        scaler.scale(total_loss).backward()

        if (batch_idx + 1) % cfg.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        # Track losses
        for k, v in loss_dict.items():
            running_loss[k] += v
        n_batches += 1

        if batch_idx % cfg.log_every_n_steps == 0:
            lr_current = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                loss=f"{loss_dict['total']:.4f}",
                lr=f"{lr_current:.6f}",
            )

    # Average losses
    avg_loss = {k: v / n_batches for k, v in running_loss.items()}
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, criterion, cfg, class_info):
    """Validate and compute per-class IoU."""
    model.eval()
    running_loss = defaultdict(float)
    n_batches = 0
    all_ious = []

    for batch in tqdm(dataloader, desc="[Validation]"):
        images = batch["image"].to(DEVICE)
        masks = batch["mask"].to(DEVICE)
        edges = batch.get("edge")
        if edges is not None:
            edges = edges.to(DEVICE)

        with autocast(enabled=cfg.use_amp):
            seg_logits, edge_logits = model(images)
            _, loss_dict = criterion(
                seg_logits, masks,
                edge_logits=edge_logits,
                edge_targets=edges,
            )

        for k, v in loss_dict.items():
            running_loss[k] += v
        n_batches += 1

        # IoU
        preds = torch.argmax(seg_logits, dim=1).cpu().numpy()
        targets = masks.cpu().numpy()
        for i in range(len(preds)):
            all_ious.append(compute_iou_per_class(preds[i], targets[i], cfg.num_classes))

    avg_loss = {k: v / n_batches for k, v in running_loss.items()}
    iou_matrix = np.array(all_ious)
    mean_iou_per_class = np.nanmean(iou_matrix, axis=0)
    miou = np.nanmean(mean_iou_per_class)

    # Print per-class results
    logger.info(f"  {'Class':<25} {'IoU':>8}")
    logger.info(f"  {'-'*35}")
    for cls_id in range(cfg.num_classes):
        cls_name = class_info.id_to_name.get(cls_id, f"class_{cls_id}")
        iou_val = mean_iou_per_class[cls_id]
        marker = " *" if iou_val < 0.1 and cls_id > 0 else ""
        logger.info(f"  {cls_name:<25} {iou_val:>8.4f}{marker}")
    logger.info(f"  {'-'*35}")
    logger.info(f"  {'MEAN IoU':<25} {miou:>8.4f}")

    return avg_loss, miou, mean_iou_per_class


# ─── Save / Load ──────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scaler, scheduler, epoch, miou, phase, out_dir, is_best=False):
    """Save training checkpoint."""
    ckpt = {
        "epoch": epoch,
        "phase": phase,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_miou": miou,
        "timestamp": datetime.now().isoformat(),
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    torch.save(ckpt, out_path / "last_model.pth")
    if is_best:
        torch.save(ckpt, out_path / "best_model.pth")
        logger.info(f"  >>> New best model saved! mIoU={miou:.4f}")


# ─── Main Training Pipeline ──────────────────────────────────────────────────

def run_training(cfg: TrainConfig):
    """Execute the full two-phase training pipeline."""
    seed_everything(cfg.seed)

    logger.info("=" * 70)
    logger.info("THIN-LINE SEGMENTATION FINE-TUNING")
    logger.info("=" * 70)
    logger.info(f"Device: {DEVICE}")

    # ── Load class info ──
    class_info = ClassInfo.from_json(cfg.classes_json)
    logger.info(f"Classes: {class_info.num_classes} (including background)")

    # ── Create model ──
    model = DeepLabV3PlusEdge(
        num_classes=cfg.num_classes,
        backbone=cfg.backbone,
        use_edge_head=cfg.use_edge_head,
    )

    # ── Load pretrained checkpoint ──
    if cfg.checkpoint_path and os.path.exists(cfg.checkpoint_path):
        stats = model.load_pretrained_checkpoint(
            cfg.checkpoint_path,
            old_num_classes=cfg.old_num_classes,
            device=DEVICE,
        )
        logger.info(f"Checkpoint loading stats: {stats}")
    else:
        logger.warning("No checkpoint found — training from ImageNet init only!")

    model = model.to(DEVICE)
    model.summary()

    # ── Datasets ──
    train_transform = get_train_transform(cfg)
    val_transform = get_val_transform()

    train_ds = PatchDataset(cfg.patch_root, split="train",
                            transform=train_transform, load_edge=cfg.use_edge_head)
    val_ds = PatchDataset(cfg.patch_root, split="val",
                          transform=val_transform, load_edge=cfg.use_edge_head)

    # Weighted sampler for class balance
    sampler = None
    shuffle = True
    if cfg.use_weighted_sampler:
        weights = train_ds.get_sampling_weights()
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False  # Sampler handles ordering

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=shuffle,
        sampler=sampler, num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    )

    # ── Class weights ──
    class_weights = compute_class_weights(train_ds, cfg.num_classes).to(DEVICE)

    # ── Loss function ──
    criterion = CombinedLoss(
        num_classes=cfg.num_classes,
        class_weights=class_weights,
        ce_weight=cfg.loss_ce_weight,
        dice_weight=cfg.loss_dice_weight,
        focal_weight=cfg.loss_focal_weight,
        edge_weight=cfg.loss_edge_weight,
        focal_gamma=cfg.focal_gamma,
        use_edge=cfg.use_edge_head,
    )

    out_dir = Path(cfg.out_dir) / "checkpoints"
    log_path = Path(cfg.out_dir) / "training_log.csv"
    log_rows = []

    best_miou = 0.0
    no_improve = 0
    global_epoch = 0
    resume_phase = None  # Which phase to resume into

    # ── Resume from checkpoint if requested ──
    resume_ckpt_path = out_dir / "last_model.pth"
    if cfg.resume and resume_ckpt_path.exists():
        logger.info(f"Resuming from: {resume_ckpt_path}")
        try:
            ckpt = torch.load(resume_ckpt_path, map_location=DEVICE, weights_only=False)
        except TypeError:
            ckpt = torch.load(resume_ckpt_path, map_location=DEVICE)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            global_epoch = ckpt.get("epoch", 0)
            best_miou = ckpt.get("best_miou", 0.0)
            resume_phase = ckpt.get("phase", "phase1")
            logger.info(f"  Resumed from epoch {global_epoch}, phase={resume_phase}, "
                         f"best_miou={best_miou:.4f}")
        else:
            model.load_state_dict(ckpt)
            logger.warning("  Legacy checkpoint — resetting epoch counter")
    elif cfg.resume:
        logger.warning(f"  Resume requested but {resume_ckpt_path} not found. Starting fresh.")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: Freeze encoder, train decoder + heads
    # ══════════════════════════════════════════════════════════════════
    # Determine if Phase 1 should be skipped (already completed)
    skip_phase1 = False
    phase1_start_epoch = 1
    if resume_phase == "phase2":
        skip_phase1 = True
        logger.info("Phase 1 already completed — skipping to Phase 2")
    elif resume_phase == "phase1" and global_epoch > 0:
        phase1_start_epoch = global_epoch - cfg.phase1_epochs + cfg.phase1_epochs  # continue
        # More precisely: resume at the next epoch
        phase1_start_epoch = min(global_epoch + 1, cfg.phase1_epochs + 1)

    if cfg.phase1_epochs > 0 and not skip_phase1:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 1: Encoder frozen, training decoder + heads")
        logger.info("=" * 70)

        model.freeze_encoder()
        model.summary()

        optimizer = build_optimizer_phase1(model, cfg)
        scaler = GradScaler(enabled=cfg.use_amp)
        scheduler = build_scheduler(
            optimizer, cfg.phase1_epochs, cfg.warmup_epochs,
            len(train_loader), mode=cfg.scheduler,
        )

        for epoch in range(phase1_start_epoch, cfg.phase1_epochs + 1):
            global_epoch += 1
            logger.info(f"\n--- Phase 1, Epoch {epoch}/{cfg.phase1_epochs} ---")

            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler,
                scaler, cfg, global_epoch, "P1",
            )

            val_loss, miou, class_ious = validate(
                model, val_loader, criterion, cfg, class_info,
            )

            logger.info(f"  Train loss: {train_loss['total']:.4f} | "
                         f"Val loss: {val_loss['total']:.4f} | Val mIoU: {miou:.4f}")

            is_best = miou > best_miou
            if is_best:
                best_miou = miou
                no_improve = 0
            else:
                no_improve += 1

            save_checkpoint(model, optimizer, scaler, scheduler,
                            global_epoch, best_miou, "phase1", out_dir, is_best)

            log_rows.append({
                "epoch": global_epoch, "phase": "phase1",
                "train_loss": train_loss["total"], "val_loss": val_loss["total"],
                "miou": miou, "best_miou": best_miou,
                "lr": optimizer.param_groups[0]["lr"],
            })

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: Unfreeze encoder, full fine-tuning with differential LR
    # ══════════════════════════════════════════════════════════════════
    if cfg.phase2_epochs > 0:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 2: Full fine-tuning with differential LR")
        logger.info("=" * 70)

        model.unfreeze_all()
        model.summary()

        optimizer = build_optimizer_phase2(model, cfg)
        scaler = GradScaler(enabled=cfg.use_amp)
        scheduler = build_scheduler(
            optimizer, cfg.phase2_epochs, cfg.warmup_epochs,
            len(train_loader), mode=cfg.scheduler,
        )

        no_improve = 0  # Reset patience for phase 2

        for epoch in range(1, cfg.phase2_epochs + 1):
            global_epoch += 1
            logger.info(f"\n--- Phase 2, Epoch {epoch}/{cfg.phase2_epochs} ---")

            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler,
                scaler, cfg, global_epoch, "P2",
            )

            val_loss, miou, class_ious = validate(
                model, val_loader, criterion, cfg, class_info,
            )

            logger.info(f"  Train loss: {train_loss['total']:.4f} | "
                         f"Val loss: {val_loss['total']:.4f} | Val mIoU: {miou:.4f}")

            is_best = miou > best_miou
            if is_best:
                best_miou = miou
                no_improve = 0
            else:
                no_improve += 1
                logger.info(f"  No improvement. Patience: {no_improve}/{cfg.early_stopping_patience}")

            save_checkpoint(model, optimizer, scaler, scheduler,
                            global_epoch, best_miou, "phase2", out_dir, is_best)

            log_rows.append({
                "epoch": global_epoch, "phase": "phase2",
                "train_loss": train_loss["total"], "val_loss": val_loss["total"],
                "miou": miou, "best_miou": best_miou,
                "lr": optimizer.param_groups[0]["lr"],
            })

            if no_improve >= cfg.early_stopping_patience:
                logger.info(f"Early stopping at epoch {global_epoch}!")
                break

    # ── Save training log ──
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    logger.info(f"\nTraining log saved: {log_path}")
    logger.info(f"Best mIoU: {best_miou:.4f}")
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DeepLabV3+ for thin-line segmentation")

    # Override config fields via CLI
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--patch_root", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--phase1_epochs", type=int, default=None)
    parser.add_argument("--phase2_epochs", type=int, default=None)
    parser.add_argument("--phase1_lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--no_edge_head", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")

    args = parser.parse_args()

    # Build config with overrides
    cfg = TrainConfig()

    if args.checkpoint_path:
        cfg.checkpoint_path = args.checkpoint_path
    if args.patch_root:
        cfg.patch_root = args.patch_root
    if args.out_dir:
        cfg.out_dir = args.out_dir
    if args.backbone:
        cfg.backbone = args.backbone
    if args.phase1_epochs is not None:
        cfg.phase1_epochs = args.phase1_epochs
    if args.phase2_epochs is not None:
        cfg.phase2_epochs = args.phase2_epochs
    if args.phase1_lr is not None:
        cfg.phase1_lr = args.phase1_lr
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.no_edge_head:
        cfg.use_edge_head = False
    if args.no_amp:
        cfg.use_amp = False
    if args.grad_accum_steps is not None:
        cfg.grad_accum_steps = args.grad_accum_steps
    cfg.resume = args.resume

    run_training(cfg)


if __name__ == "__main__":
    main()
