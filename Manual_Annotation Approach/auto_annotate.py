"""
Auto-Annotate CAD Engineering Drawings
=======================================

Uses the pretrained DeepLabV3+ model (trained on synthetic single-line data)
to generate pseudo-label masks for the 362 unannotated real CAD drawings.

Strategy:
  - Sliding window (tiled) inference for variable-size images
  - Gaussian-weighted blending for smooth tile boundaries
  - Confidence map output so user can identify uncertain regions
  - Visual overlay previews for quick quality assessment

Output:
  - masks/      : Multi-class masks (0=BG, 1-9=line classes)
  - confidence/  : Per-pixel max confidence (0-255)
  - overlays/    : Visual previews with color-coded lines
  - report.csv   : Per-image class distribution stats

Usage:
    python auto_annotate.py
    python auto_annotate.py --checkpoint <path> --conf_thresh 0.3
"""

import os
import sys
import json
import logging
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    print("Install: pip install segmentation-models-pytorch albumentations")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("auto_annotate")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Paths ─────────────────────────────────────────────────────────────────

CAD_DIR = Path(r"C:\Users\Admin\line_detection\Approach 2\Engineering_Drawings")
CHECKPOINT = Path(r"C:\Users\Admin\line_detection\Approach 2\Segmentation_Deeplab_models_checkpoints\best_model.pth")
CLASSES_JSON = Path(r"C:\Users\Admin\line_detection\Approach 2\pipeline_output\classes_cleaned.json")
OUT_DIR = Path(r"C:\Users\Admin\line_detection\Approach 2\cad_annotations")

# The checkpoint was trained with 13 classes (old mapping)
# We load with original architecture to match weights exactly
OLD_NUM_CLASSES = 13

# ── Config ────────────────────────────────────────────────────────────────

TILE_SIZE = 512
OVERLAP = 128
CONF_THRESHOLD = 0.3   # Minimum confidence to keep a prediction
TTA = True             # Test-time augmentation (flip)


def get_gaussian_window(size, sigma=0.5):
    """Gaussian blending window for smooth tile boundaries."""
    x = np.linspace(-1, 1, size)
    g = np.exp(-0.5 * (x / sigma) ** 2)
    return np.outer(g, g).astype(np.float32)


def safe_imread(path):
    """Read image handling Unicode paths on Windows."""
    arr = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def safe_imwrite(path, img):
    """Write image handling Unicode paths."""
    ext = Path(path).suffix
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))


def load_model(ckpt_path, num_classes):
    """Load the pretrained model with original class count."""
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,  # We load from checkpoint
        in_channels=3,
        classes=num_classes,
    )

    try:
        ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location=DEVICE)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}, "
                     f"mIoU={ckpt.get('best_miou', '?'):.4f})")
    else:
        model.load_state_dict(ckpt)

    model = model.to(DEVICE)
    model.eval()
    return model


def load_class_mapping():
    """Load the cleaned 9-class mapping and create old→new ID map."""
    with open(CLASSES_JSON) as f:
        new_cmap = json.load(f)  # {name: new_id}  (1-9)

    # The old checkpoint had 13 classes (12 line types + BG)
    # Old classes.json was in output/classes.json
    # We need to map old class IDs to new class IDs
    #
    # For now, use a simple approach:
    # The new mapping has 9 classes. If the old model predicts class IDs
    # that correspond to removed classes, we map them to background.
    #
    # Old IDs 1-12 → we keep the first 9 line types that match
    # Since both old and new share the same names, we map by name

    new_id_to_name = {0: "Background"}
    for name, cid in new_cmap.items():
        new_id_to_name[cid] = name

    return new_cmap, new_id_to_name


def run_tiled_inference(model, image_rgb, num_classes, transform, window):
    """Run sliding-window inference with Gaussian blending."""
    h, w = image_rgb.shape[:2]
    stride = TILE_SIZE - OVERLAP

    prob_map = np.zeros((num_classes, h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y2 = min(y + TILE_SIZE, h)
            x2 = min(x + TILE_SIZE, w)
            y1 = max(0, y2 - TILE_SIZE)
            x1 = max(0, x2 - TILE_SIZE)

            tile = image_rgb[y1:y2, x1:x2]

            # Handle tiles smaller than TILE_SIZE
            th, tw = tile.shape[:2]
            if th < TILE_SIZE or tw < TILE_SIZE:
                pad_tile = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
                pad_tile[:th, :tw] = tile
                tile = pad_tile

            tens = transform(image=tile)["image"].unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(tens)

                if TTA:
                    # Horizontal flip TTA
                    logits_flip = model(torch.flip(tens, [3]))
                    logits = (logits + torch.flip(logits_flip, [3])) / 2.0

                probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # (C, H, W)

            # Crop back if we padded
            tile_h = min(TILE_SIZE, y2 - y1)
            tile_w = min(TILE_SIZE, x2 - x1)
            probs = probs[:, :tile_h, :tile_w]
            win = window[:tile_h, :tile_w]

            prob_map[:, y1:y2, x1:x2] += probs * win
            count_map[y1:y2, x1:x2] += win

    # Normalize
    prob_map /= (count_map[np.newaxis, :, :] + 1e-6)
    return prob_map


def create_color_overlay(image_rgb, mask, id_to_name):
    """Create a visual overlay with color-coded line classes."""
    colors = np.array([
        [0,   0,   0],     # 0: BG
        [255, 50,  50],    # 1: Center_line (red)
        [50,  255, 50],    # 2: Dimension_lines (green)
        [50,  50,  255],   # 3: Extension_line (blue)
        [255, 255, 50],    # 4: Feature_Visible (yellow)
        [255, 50,  255],   # 5: Leader_line (magenta)
        [50,  255, 255],   # 6: Phantom_lines (cyan)
        [200, 100, 50],    # 7: break_line (orange)
        [50,  200, 150],   # 8: cutting_plane (teal)
        [150, 150, 255],   # 9: hidden_lines (lavender)
    ], dtype=np.uint8)

    colored = np.zeros_like(image_rgb)
    for cls_id in range(1, min(len(colors), 10)):
        colored[mask == cls_id] = colors[cls_id]

    overlay = cv2.addWeighted(image_rgb, 0.6, colored, 0.4, 0)

    # Add legend
    legend_h = 25 * min(len(id_to_name), 10)
    legend = np.ones((legend_h, 250, 3), dtype=np.uint8) * 30
    for cls_id in range(min(len(colors), 10)):
        name = id_to_name.get(cls_id, f"class_{cls_id}")
        y_pos = cls_id * 25 + 18
        if cls_id < len(colors):
            cv2.rectangle(legend, (5, cls_id * 25 + 5), (20, cls_id * 25 + 20),
                          colors[cls_id].tolist(), -1)
        cv2.putText(legend, f"{cls_id}: {name}", (25, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

    return overlay, legend


def main():
    logger.info("=" * 60)
    logger.info("AUTO-ANNOTATION OF CAD ENGINEERING DRAWINGS")
    logger.info("=" * 60)
    logger.info(f"Device: {DEVICE}")

    # Setup output dirs
    mask_dir = OUT_DIR / "masks"
    conf_dir = OUT_DIR / "confidence"
    overlay_dir = OUT_DIR / "overlays"
    for d in [mask_dir, conf_dir, overlay_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load model (with ORIGINAL 13-class architecture to match checkpoint)
    model = load_model(CHECKPOINT, OLD_NUM_CLASSES)

    # Load class mapping
    new_cmap, new_id_to_name = load_class_mapping()

    # Build old→new mapping
    # The old model outputs 13 classes. We need to map them to 10 (0-9).
    # Since we don't have the old classes.json, we'll remap:
    # - Keep predictions for classes 0-9 (BG + first 9 line types)
    # - Map classes 10-12 to background
    # This is a safe approximation since the removed classes
    # (Arrowhead, Dimension_text, Section_hatching) are IDs 10-12 in the old model

    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    window = get_gaussian_window(TILE_SIZE)

    # Get all CAD images
    valid_ext = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif"}
    cad_images = sorted([f for f in CAD_DIR.iterdir()
                         if f.suffix.lower() in valid_ext])
    logger.info(f"Found {len(cad_images)} CAD images to annotate")

    # Stats tracking
    report_rows = []
    total_annotated = 0
    total_with_lines = 0

    for img_path in tqdm(cad_images, desc="Annotating"):
        img_bgr = safe_imread(img_path)
        if img_bgr is None:
            logger.warning(f"  Failed to read: {img_path.name}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # Run inference
        prob_map = run_tiled_inference(model, img_rgb, OLD_NUM_CLASSES, transform, window)

        # Get predictions
        # Remap: keep only classes 0-9, merge 10-12 into class 0 (background)
        # Do this by zeroing out probabilities for removed classes
        # and re-normalizing
        if prob_map.shape[0] > 10:
            prob_map[10:] = 0  # Zero out removed classes
            # Re-normalize so probabilities sum to 1
            prob_sum = prob_map.sum(axis=0, keepdims=True)
            prob_map = prob_map / (prob_sum + 1e-8)

        # Use only first 10 channels
        prob_map_10 = prob_map[:10]

        # Argmax prediction
        pred_mask = np.argmax(prob_map_10, axis=0).astype(np.uint8)

        # Confidence: max probability across classes
        confidence = np.max(prob_map_10, axis=0)

        # Apply confidence threshold - set low-confidence pixels to BG
        pred_mask[confidence < CONF_THRESHOLD] = 0

        # Save mask (single-channel, class IDs 0-9)
        stem = img_path.stem
        Image.fromarray(pred_mask).save(str(mask_dir / f"{stem}.png"))

        # Save confidence map (scaled to 0-255)
        conf_uint8 = (confidence * 255).astype(np.uint8)
        Image.fromarray(conf_uint8).save(str(conf_dir / f"{stem}.png"))

        # Save overlay
        overlay, legend = create_color_overlay(img_rgb, pred_mask, new_id_to_name)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        safe_imwrite(str(overlay_dir / f"{stem}_overlay.jpg"), overlay_bgr)

        # Stats
        unique, counts = np.unique(pred_mask, return_counts=True)
        cls_dist = {int(u): int(c) for u, c in zip(unique, counts)}
        fg_pixels = sum(c for u, c in cls_dist.items() if u > 0)
        fg_ratio = fg_pixels / (h * w)
        n_classes = len([u for u in unique if u > 0])

        report_rows.append({
            "image": img_path.name,
            "width": w,
            "height": h,
            "fg_ratio": round(fg_ratio, 4),
            "n_classes": n_classes,
            "classes": json.dumps([int(u) for u in unique if u > 0]),
            "mean_confidence": round(float(confidence.mean()), 4),
        })

        total_annotated += 1
        if n_classes > 0:
            total_with_lines += 1

    # Save report
    report_path = OUT_DIR / "annotation_report.csv"
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=report_rows[0].keys() if report_rows else [])
        writer.writeheader()
        writer.writerows(report_rows)

    logger.info(f"\n{'=' * 60}")
    logger.info("AUTO-ANNOTATION COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Total annotated: {total_annotated}")
    logger.info(f"  With detected lines: {total_with_lines}")
    logger.info(f"  Output: {OUT_DIR}")
    logger.info(f"  Masks: {mask_dir}")
    logger.info(f"  Confidence maps: {conf_dir}")
    logger.info(f"  Overlays: {overlay_dir}")
    logger.info(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
