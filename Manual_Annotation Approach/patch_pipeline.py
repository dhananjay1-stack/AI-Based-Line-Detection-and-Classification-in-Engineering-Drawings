#!/usr/bin/env python3
"""
Patch Extraction Pipeline for Engineering Drawing Thin-Line Segmentation
========================================================================

Extracts balanced, foreground-rich patches from the cleaned dataset,
generates centerline/skeleton targets, and produces metadata for
class-balanced training.

Usage:
    python patch_pipeline.py extract   [--patch_size 512] [--stride 384] ...
    python patch_pipeline.py preview   [--n_samples 20]
    python patch_pipeline.py run_all   [--patch_size 512] ...

Requirements:
    pip install numpy opencv-python Pillow tqdm pandas scikit-image
"""

import os
import sys
import json
import argparse
import logging
import random
import csv
from pathlib import Path
from collections import defaultdict, OrderedDict, Counter
from datetime import datetime

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import pandas as pd

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("patch_pipeline")

# =============================================================================
# Constants
# =============================================================================
SEED = 42
NUM_CLASSES = 9  # IDs 1..9, plus 0=background


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


# =============================================================================
# Image I/O Helpers (handle Unicode paths on Windows)
# =============================================================================

def safe_imread(path, flags=cv2.IMREAD_UNCHANGED):
    try:
        arr = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(arr, flags)
    except Exception:
        return None


def safe_imwrite(path, img):
    try:
        ext = Path(path).suffix
        ok, buf = cv2.imencode(ext, img)
        if ok:
            buf.tofile(str(path))
            return True
    except Exception:
        pass
    return False


# =============================================================================
# Skeletonization
# =============================================================================

def compute_skeleton(mask):
    """
    Compute 1-pixel skeleton from a binary or multi-class mask.
    Returns a multi-class skeleton mask where each foreground pixel
    retains its original class ID but is thinned to 1px width.
    """
    skeleton = np.zeros_like(mask, dtype=np.uint8)
    unique_classes = np.unique(mask)

    for cls_id in unique_classes:
        if cls_id == 0:
            continue
        binary = (mask == cls_id).astype(np.uint8)

        # Morphological thinning via iterative erosion
        # Using Zhang-Suen via OpenCV ximgproc if available, else manual
        try:
            thinned = cv2.ximgproc.thinning(binary * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
            thinned = (thinned > 0).astype(np.uint8)
        except AttributeError:
            # Fallback: use skimage skeletonize
            try:
                from skimage.morphology import skeletonize
                thinned = skeletonize(binary).astype(np.uint8)
            except ImportError:
                # Last resort: simple iterative morphological thinning
                thinned = _manual_thin(binary)

        skeleton[thinned > 0] = cls_id

    return skeleton


def _manual_thin(binary):
    """Simple morphological skeleton as fallback."""
    skel = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = binary.copy()
    while True:
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


# =============================================================================
# Patch Scoring / Richness Analysis
# =============================================================================

def compute_patch_score(mask_patch, class_frequencies):
    """
    Score a patch based on how useful it is for training.
    Higher score = more valuable patch.

    Factors:
    - Foreground ratio (more line pixels = better)
    - Number of distinct classes (multi-class patches are gold)
    - Presence of rare classes (boost rare classes)
    - Spatial complexity (line density variation = intersections)
    """
    h, w = mask_patch.shape
    total_px = h * w

    # 1. Foreground ratio
    fg_px = np.count_nonzero(mask_patch)
    fg_ratio = fg_px / total_px

    if fg_ratio < 1e-6:
        return 0.0, {
            "fg_ratio": 0.0,
            "n_classes": 0,
            "has_rare": False,
            "complexity": 0.0,
            "classes_present": [],
        }

    # 2. Class diversity
    unique_classes = [c for c in np.unique(mask_patch) if c > 0]
    n_classes = len(unique_classes)

    # 3. Rare class boost
    # class_frequencies: dict {cls_id: total_pixel_count}
    total_all = sum(class_frequencies.values()) + 1
    rarity_score = 0.0
    has_rare = False
    for cls_id in unique_classes:
        cls_freq = class_frequencies.get(cls_id, 1) / total_all
        # Inverse frequency weight, capped
        inv_freq = min(1.0 / (cls_freq + 1e-6), 100.0)
        rarity_score += inv_freq
        if cls_freq < 0.05:  # Class occupies < 5% of total pixels
            has_rare = True

    rarity_score = rarity_score / max(n_classes, 1)

    # 4. Spatial complexity: measure line density variations
    # Use gradient magnitude of the mask as a proxy for intersections
    binary_fg = (mask_patch > 0).astype(np.uint8)
    gx = cv2.Sobel(binary_fg, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(binary_fg, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    complexity = float(np.mean(grad_mag))

    # Combined score
    score = (
        fg_ratio * 10.0           # Base: more foreground = better
        + n_classes * 5.0          # Multi-class bonus
        + rarity_score * 0.5       # Rare class presence
        + complexity * 20.0        # Intersection/edge density
    )

    info = {
        "fg_ratio": round(fg_ratio, 6),
        "n_classes": n_classes,
        "has_rare": has_rare,
        "complexity": round(complexity, 6),
        "classes_present": [int(c) for c in unique_classes],
    }

    return score, info


# =============================================================================
# Phase 1: EXTRACT PATCHES
# =============================================================================

def run_extract(args):
    """
    Extract patches from cleaned dataset with intelligent sampling:
    - Sliding window with overlap
    - Foreground threshold to reject empty patches
    - Scoring to prioritize line-rich regions
    - Oversampling of rare-class patches
    - Skeleton/centerline generation
    """
    logger.info("=" * 70)
    logger.info("PATCH EXTRACTION PIPELINE")
    logger.info("=" * 70)

    set_seed(SEED)

    # --- Config ---
    PATCH_SIZE = args.patch_size
    STRIDE = args.stride
    FG_THRESHOLD = args.fg_threshold
    MAX_PATCHES_PER_IMAGE = args.max_patches_per_image
    RARE_OVERSAMPLE = args.rare_oversample

    # --- Paths ---
    pipeline_root = Path(args.pipeline_output)
    img_dir = pipeline_root / "cleaned" / "images"
    mask_dir = pipeline_root / "cleaned" / "masks"
    classes_path = pipeline_root / "classes_cleaned.json"
    splits_dir = pipeline_root / "splits"

    out_root = Path(args.out_dir)
    out_images = out_root / "images"
    out_masks = out_root / "masks"
    out_skeletons = out_root / "centerlines"
    out_meta = out_root / "metadata"
    out_preview = out_root / "preview"

    for d in [out_images, out_masks, out_skeletons, out_meta, out_preview]:
        d.mkdir(parents=True, exist_ok=True)

    # --- Load class map ---
    with open(classes_path) as f:
        class_map = json.load(f)
    id_to_class = {v: k for k, v in class_map.items()}

    logger.info(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}, Stride: {STRIDE}")
    logger.info(f"FG threshold: {FG_THRESHOLD}, Max patches/image: {MAX_PATCHES_PER_IMAGE}")
    logger.info(f"Rare class oversample factor: {RARE_OVERSAMPLE}")

    # --- First pass: compute global class pixel frequencies ---
    logger.info("Computing global class frequencies for rarity scoring...")
    class_pixel_counts = defaultdict(int)

    # Sample a subset for frequency estimation
    all_masks = sorted(mask_dir.glob("*.png"))
    sample_for_freq = random.sample(all_masks, min(2000, len(all_masks)))

    for mp in tqdm(sample_for_freq, desc="Estimating class frequencies"):
        m = np.array(Image.open(mp))
        unique, counts = np.unique(m, return_counts=True)
        for u, c in zip(unique, counts):
            if u > 0:
                class_pixel_counts[int(u)] += int(c)

    logger.info("Class pixel frequencies (estimated):")
    for cls_id in sorted(class_pixel_counts.keys()):
        cls_name = id_to_class.get(cls_id, f"class_{cls_id}")
        logger.info(f"  {cls_id}: {cls_name} = {class_pixel_counts[cls_id]:,} px")

    # --- Process each split ---
    all_metadata = []
    global_stats = {
        "total_patches": 0,
        "rejected_low_fg": 0,
        "rejected_cap": 0,
        "oversampled_rare": 0,
        "class_patch_counts": defaultdict(int),
        "split_counts": {},
    }

    for split_name in ["train", "val", "test"]:
        split_file = splits_dir / f"{split_name}.txt"
        if not split_file.exists():
            logger.warning(f"Split file not found: {split_file}")
            continue

        with open(split_file) as f:
            stems = [line.strip() for line in f if line.strip()]

        logger.info(f"\n--- Processing {split_name} split: {len(stems)} images ---")

        split_patches = 0
        split_rejected_fg = 0
        split_rejected_cap = 0

        for stem in tqdm(stems, desc=f"Extracting [{split_name}]"):
            img_path = img_dir / f"{stem}.png"
            mask_path = mask_dir / f"{stem}.png"

            if not img_path.exists() or not mask_path.exists():
                continue

            img = safe_imread(str(img_path))
            mask = np.array(Image.open(mask_path))

            if img is None:
                continue

            # Ensure 3-channel
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            h, w = img.shape[:2]

            # Collect candidate patches with scores
            candidates = []

            for y in range(0, h, STRIDE):
                for x in range(0, w, STRIDE):
                    # Compute crop region with padding at borders
                    y1, x1 = y, x
                    y2 = min(y + PATCH_SIZE, h)
                    x2 = min(x + PATCH_SIZE, w)

                    # Shift back to maintain patch size at borders
                    if y2 - y1 < PATCH_SIZE and h >= PATCH_SIZE:
                        y1 = h - PATCH_SIZE
                        y2 = h
                    if x2 - x1 < PATCH_SIZE and w >= PATCH_SIZE:
                        x1 = w - PATCH_SIZE
                        x2 = w

                    img_patch = img[y1:y2, x1:x2]
                    mask_patch = mask[y1:y2, x1:x2]

                    # Pad if image is smaller than patch size
                    ph, pw = mask_patch.shape[:2]
                    if ph < PATCH_SIZE or pw < PATCH_SIZE:
                        pad_h = PATCH_SIZE - ph
                        pad_w = PATCH_SIZE - pw
                        img_patch = np.pad(img_patch, ((0, pad_h), (0, pad_w), (0, 0)),
                                           mode="constant", constant_values=0)
                        mask_patch = np.pad(mask_patch, ((0, pad_h), (0, pad_w)),
                                            mode="constant", constant_values=0)

                    # Score the patch
                    score, info = compute_patch_score(mask_patch, class_pixel_counts)

                    # Foreground threshold filter
                    if info["fg_ratio"] < FG_THRESHOLD:
                        split_rejected_fg += 1
                        continue

                    candidates.append({
                        "y1": y1, "x1": x1, "y2": y2, "x2": x2,
                        "score": score,
                        "info": info,
                        "img_patch": img_patch,
                        "mask_patch": mask_patch,
                    })

            # If no valid patches, try the full image center crop
            if not candidates:
                cy = max(0, (h - PATCH_SIZE) // 2)
                cx = max(0, (w - PATCH_SIZE) // 2)
                img_patch = img[cy:cy+PATCH_SIZE, cx:cx+PATCH_SIZE]
                mask_patch = mask[cy:cy+PATCH_SIZE, cx:cx+PATCH_SIZE]
                ph, pw = mask_patch.shape[:2]
                if ph < PATCH_SIZE or pw < PATCH_SIZE:
                    img_patch = np.pad(img_patch,
                                       ((0, PATCH_SIZE-ph), (0, PATCH_SIZE-pw), (0, 0)),
                                       mode="constant")
                    mask_patch = np.pad(mask_patch,
                                        ((0, PATCH_SIZE-ph), (0, PATCH_SIZE-pw)),
                                        mode="constant")
                score, info = compute_patch_score(mask_patch, class_pixel_counts)
                candidates.append({
                    "y1": cy, "x1": cx,
                    "y2": cy + PATCH_SIZE, "x2": cx + PATCH_SIZE,
                    "score": score, "info": info,
                    "img_patch": img_patch, "mask_patch": mask_patch,
                })

            # Sort by score, keep ONLY the single best patch per image
            candidates.sort(key=lambda c: c["score"], reverse=True)
            best = candidates[0]
            split_rejected_cap += len(candidates) - 1

            # Compute sampling weight for class-balanced training
            # Rare classes get higher weight so dataloader can oversample them
            sampling_weight = 1.0
            if best["info"]["has_rare"] and split_name == "train":
                sampling_weight = float(RARE_OVERSAMPLE)
                global_stats["oversampled_rare"] += 1

            patch_name = f"{stem}_p00"

            # Save image patch as JPEG (saves ~70% space vs PNG)
            safe_imwrite(str(out_images / f"{patch_name}.jpg"), best["img_patch"])

            # Save mask patch as lossless PNG (class IDs must be exact)
            Image.fromarray(best["mask_patch"]).save(
                str(out_masks / f"{patch_name}.png")
            )

            # Compute and save skeleton
            skel = compute_skeleton(best["mask_patch"])
            Image.fromarray(skel).save(
                str(out_skeletons / f"{patch_name}.png")
            )

            # Build metadata row
            meta = {
                "patch_name": patch_name,
                "split": split_name,
                "source_image": stem,
                "y1": best["y1"],
                "x1": best["x1"],
                "y2": best["y2"],
                "x2": best["x2"],
                "score": round(best["score"], 4),
                "fg_ratio": best["info"]["fg_ratio"],
                "n_classes": best["info"]["n_classes"],
                "has_rare": best["info"]["has_rare"],
                "complexity": best["info"]["complexity"],
                "classes_present": json.dumps(best["info"]["classes_present"]),
                "sampling_weight": sampling_weight,
            }
            all_metadata.append(meta)
            split_patches += 1

            # Track class patch counts
            for cls_id in best["info"]["classes_present"]:
                global_stats["class_patch_counts"][cls_id] += 1

        global_stats["split_counts"][split_name] = split_patches
        global_stats["total_patches"] += split_patches
        global_stats["rejected_low_fg"] += split_rejected_fg
        global_stats["rejected_cap"] += split_rejected_cap

        logger.info(f"  {split_name}: {split_patches} patches extracted, "
                     f"{split_rejected_fg} rejected (low FG), "
                     f"{split_rejected_cap} capped")

    # --- Save metadata CSV ---
    meta_df = pd.DataFrame(all_metadata)
    meta_csv_path = out_meta / "patch_metadata.csv"
    meta_df.to_csv(meta_csv_path, index=False)
    logger.info(f"Metadata saved: {meta_csv_path} ({len(meta_df)} rows)")

    # --- Save split files for patches ---
    for split_name in ["train", "val", "test"]:
        split_patches = meta_df[meta_df["split"] == split_name]["patch_name"].tolist()
        split_path = out_meta / f"{split_name}_patches.txt"
        with open(split_path, "w") as f:
            f.write("\n".join(split_patches) + "\n")
        logger.info(f"  {split_name} patch list: {len(split_patches)} → {split_path}")

    # --- Save class distribution ---
    dist_rows = []
    for cls_id in sorted(global_stats["class_patch_counts"].keys()):
        cls_name = id_to_class.get(cls_id, f"class_{cls_id}")
        count = global_stats["class_patch_counts"][cls_id]
        dist_rows.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "patch_count": count,
        })

    dist_df = pd.DataFrame(dist_rows)
    dist_path = out_meta / "class_distribution_patches.csv"
    dist_df.to_csv(dist_path, index=False)

    # --- Save summary report ---
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "patch_size": PATCH_SIZE,
            "stride": STRIDE,
            "fg_threshold": FG_THRESHOLD,
            "max_patches_per_image": MAX_PATCHES_PER_IMAGE,
            "rare_oversample": RARE_OVERSAMPLE,
            "seed": SEED,
        },
        "results": {
            "total_patches": global_stats["total_patches"],
            "rejected_low_fg": global_stats["rejected_low_fg"],
            "rejected_cap": global_stats["rejected_cap"],
            "oversampled_rare": global_stats["oversampled_rare"],
            "split_counts": global_stats["split_counts"],
        },
        "class_distribution": dist_rows,
    }

    summary_path = out_meta / "extraction_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary: {summary_path}")

    # --- Print final summary ---
    print("\n" + "=" * 70)
    print("PATCH EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}, Stride: {STRIDE}")
    print(f"FG threshold: {FG_THRESHOLD}")
    print(f"Total patches: {global_stats['total_patches']}")
    print(f"  Train: {global_stats['split_counts'].get('train', 0)}")
    print(f"  Val:   {global_stats['split_counts'].get('val', 0)}")
    print(f"  Test:  {global_stats['split_counts'].get('test', 0)}")
    print(f"Rejected (low FG): {global_stats['rejected_low_fg']}")
    print(f"Rejected (cap):    {global_stats['rejected_cap']}")
    print(f"Oversampled rare:  {global_stats['oversampled_rare']}")
    print("-" * 70)
    print(f"{'Class':<25} {'ID':>3} {'Patches':>8}")
    print("-" * 70)
    for row in dist_rows:
        print(f"{row['class_name']:<25} {row['class_id']:>3} {row['patch_count']:>8}")
    print("=" * 70)


def _random_augment(img, mask):
    """Apply simple spatial augmentation for oversampled patches."""
    aug_type = random.choice(["flip_h", "flip_v", "rot90", "rot180", "rot270"])

    if aug_type == "flip_h":
        img = np.fliplr(img).copy()
        mask = np.fliplr(mask).copy()
    elif aug_type == "flip_v":
        img = np.flipud(img).copy()
        mask = np.flipud(mask).copy()
    elif aug_type == "rot90":
        img = np.rot90(img, 1).copy()
        mask = np.rot90(mask, 1).copy()
    elif aug_type == "rot180":
        img = np.rot90(img, 2).copy()
        mask = np.rot90(mask, 2).copy()
    elif aug_type == "rot270":
        img = np.rot90(img, 3).copy()
        mask = np.rot90(mask, 3).copy()

    return img, mask


# =============================================================================
# Phase 2: PREVIEW
# =============================================================================

def run_preview(args):
    """Generate visual previews of extracted patches."""
    logger.info("=" * 70)
    logger.info("PATCH PREVIEW GENERATION")
    logger.info("=" * 70)

    set_seed(SEED)

    out_root = Path(args.out_dir)
    img_dir = out_root / "images"
    mask_dir = out_root / "masks"
    skel_dir = out_root / "centerlines"
    meta_dir = out_root / "metadata"
    preview_dir = out_root / "preview"
    preview_dir.mkdir(exist_ok=True)

    # Load class map
    pipeline_root = Path(args.pipeline_output)
    classes_path = pipeline_root / "classes_cleaned.json"
    with open(classes_path) as f:
        class_map = json.load(f)
    id_to_class = {v: k for k, v in class_map.items()}

    # Color palette
    colors = np.array([
        [0,   0,   0],     # 0: BG
        [255, 0,   0],     # 1: Center_line
        [0,   255, 0],     # 2: Dimension_lines
        [0,   0,   255],   # 3: Extension_line
        [255, 255, 0],     # 4: Feature_Visible
        [255, 0,   255],   # 5: Leader_line
        [0,   255, 255],   # 6: Phantom_lines
        [128, 0,   255],   # 7: break_line
        [0,   128, 128],   # 8: cutting_plane
        [128, 128, 0],     # 9: hidden_lines
    ], dtype=np.uint8)

    # Load metadata
    meta_path = meta_dir / "patch_metadata.csv"
    if not meta_path.exists():
        logger.error(f"Metadata not found: {meta_path}. Run 'extract' first.")
        return

    df = pd.read_csv(meta_path)

    # Sample patches for each split
    n = args.n_samples

    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        if len(split_df) == 0:
            continue

        samples = split_df.sample(n=min(n, len(split_df)), random_state=SEED)

        for i, (_, row) in enumerate(samples.iterrows()):
            pname = row["patch_name"]

            # Images are saved as JPEG, masks and skeletons as PNG
            img_path = img_dir / f"{pname}.jpg"
            mask_path = mask_dir / f"{pname}.png"
            skel_path = skel_dir / f"{pname}.png"

            if not img_path.exists():
                continue

            img = safe_imread(str(img_path))
            mask = np.array(Image.open(mask_path)) if mask_path.exists() else None
            skel = np.array(Image.open(skel_path)) if skel_path.exists() else None

            if img is None:
                continue

            # Ensure BGR for display
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            h, w = img.shape[:2]

            # Create color mask overlay
            color_mask = np.zeros_like(img)
            if mask is not None:
                for cls_id in range(1, len(colors)):
                    if cls_id < len(colors):
                        color_mask[mask == cls_id] = colors[cls_id][::-1]  # RGB→BGR

            overlay = cv2.addWeighted(img, 0.6, color_mask, 0.4, 0)

            # Create skeleton overlay
            skel_vis = img.copy()
            if skel is not None:
                for cls_id in range(1, len(colors)):
                    if cls_id < len(colors):
                        skel_vis[skel == cls_id] = colors[cls_id][::-1]

            # Title bar
            title_h = 50
            panel_w = w * 3
            title_bar = np.ones((title_h, panel_w, 3), dtype=np.uint8) * 30

            info_text = (f"[{split_name.upper()}] {pname}  |  "
                         f"FG={row['fg_ratio']:.4f}  Score={row['score']:.2f}  "
                         f"Classes={row['classes_present']}")
            cv2.putText(title_bar, info_text, (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Labels
            label_h = 25
            labels = np.ones((label_h, panel_w, 3), dtype=np.uint8) * 50
            cv2.putText(labels, "Original", (w // 3, 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(labels, "Mask Overlay", (w + w // 3, 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(labels, "Skeleton", (2 * w + w // 3, 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Combine panels
            panels = np.hstack([img, overlay, skel_vis])
            combined = np.vstack([title_bar, labels, panels])

            out_path = preview_dir / f"{split_name}_patch_{i:03d}.png"
            safe_imwrite(str(out_path), combined)

        logger.info(f"  {split_name}: {min(n, len(split_df))} previews → {preview_dir}")

    logger.info(f"Preview images saved to: {preview_dir}")


# =============================================================================
# Phase 3: RUN ALL
# =============================================================================

def run_all(args):
    """Run extract + preview sequentially."""
    logger.info("Running full pipeline: extract → preview")
    run_extract(args)
    run_preview(args)
    logger.info("=" * 70)
    logger.info("PATCH PIPELINE COMPLETE")
    logger.info("=" * 70)


# =============================================================================
# CLI
# =============================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="Patch Extraction Pipeline for Thin-Line Segmentation"
    )
    subs = parser.add_subparsers(dest="mode", required=True)

    # Shared
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--pipeline_output", type=str,
        default=r"C:\Users\Admin\line_detection\Approach 2\pipeline_output",
        help="Path to pipeline_output from dataset_pipeline.py"
    )
    parent.add_argument(
        "--out_dir", type=str,
        default=r"C:\Users\Admin\line_detection\Approach 2\dataset_patches",
        help="Output directory for patches"
    )
    parent.add_argument("--verbose", action="store_true")

    # Extract
    p_ext = subs.add_parser("extract", parents=[parent])
    p_ext.add_argument("--patch_size", type=int, default=512)
    p_ext.add_argument("--stride", type=int, default=384,
                       help="Stride for sliding window (smaller = more overlap)")
    p_ext.add_argument("--fg_threshold", type=float, default=0.0005,
                       help="Min foreground ratio to keep a patch (0.05%%)")
    p_ext.add_argument("--max_patches_per_image", type=int, default=6,
                       help="Max patches to keep per source image (top scored)")
    p_ext.add_argument("--rare_oversample", type=int, default=2,
                       help="Oversample factor for rare-class patches")

    # Preview
    p_prev = subs.add_parser("preview", parents=[parent])
    p_prev.add_argument("--n_samples", type=int, default=20)

    # Run all
    p_all = subs.add_parser("run_all", parents=[parent])
    p_all.add_argument("--patch_size", type=int, default=512)
    p_all.add_argument("--stride", type=int, default=384)
    p_all.add_argument("--fg_threshold", type=float, default=0.0005)
    p_all.add_argument("--max_patches_per_image", type=int, default=6)
    p_all.add_argument("--rare_oversample", type=int, default=2)
    p_all.add_argument("--n_samples", type=int, default=20)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if hasattr(args, "verbose") and args.verbose:
        logger.setLevel(logging.DEBUG)

    {"extract": run_extract, "preview": run_preview, "run_all": run_all}[args.mode](args)
