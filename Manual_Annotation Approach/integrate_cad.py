"""
Integrate CAD Annotations into Training Pipeline
=================================================

Takes the auto-annotated CAD drawings and:
  1. Extracts 512x512 patches (same as synthetic pipeline)
  2. Generates skeleton/centerline targets
  3. Merges with existing synthetic patch dataset
  4. Creates a new combined metadata CSV with proper sampling weights

The combined dataset is written to dataset_patches_combined/ so
the original synthetic dataset remains untouched.

Usage:
    python integrate_cad.py                  # Run integration
    python integrate_cad.py --preview_only   # Just show stats, no patching
"""

import os
import sys
import json
import shutil
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    from skimage.morphology import skeletonize
except ImportError:
    print("Install: pip install scikit-image")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("integrate_cad")

# ── Paths ─────────────────────────────────────────────────────────────────

CAD_IMG_DIR = Path(r"C:\Users\Admin\line_detection\Approach 2\Engineering_Drawings")
CAD_MASK_DIR = Path(r"C:\Users\Admin\line_detection\Approach 2\cad_annotations\masks")
CAD_REPORT = Path(r"C:\Users\Admin\line_detection\Approach 2\cad_annotations\annotation_report.csv")
CLASSES_JSON = Path(r"C:\Users\Admin\line_detection\Approach 2\pipeline_output\classes_cleaned.json")

# Existing synthetic patches
SYNTH_PATCH_DIR = Path(r"C:\Users\Admin\line_detection\Approach 2\dataset_patches")

# Output: combined dataset
COMBINED_DIR = Path(r"C:\Users\Admin\line_detection\Approach 2\dataset_patches_combined")

# ── Config ────────────────────────────────────────────────────────────────

PATCH_SIZE = 512
STRIDE = 384          # 128px overlap for dense coverage
MIN_FG_RATIO = 0.005  # At least 0.5% foreground in a patch to keep it
MAX_PATCHES_PER_IMAGE = 8  # Cap patches per CAD image (they're large)
TRAIN_RATIO = 0.85    # 85% of CAD patches go to train, 15% to val
SEED = 42
NUM_CLASSES = 10


def safe_imread(path):
    """Read image handling Unicode paths on Windows."""
    arr = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def compute_patch_score(mask_patch, num_classes=NUM_CLASSES):
    """Score a patch by foreground richness (same logic as patch_pipeline.py)."""
    total = mask_patch.size
    if total == 0:
        return 0.0

    fg_mask = mask_patch > 0
    fg_ratio = fg_mask.sum() / total

    unique_classes = set(np.unique(mask_patch)) - {0}
    class_diversity = len(unique_classes) / max(1, num_classes - 1)

    # Bonus for rare classes
    rare_classes = {1, 5, 6, 8}  # Center_line, Leader_line, Phantom, cutting_plane
    rare_count = len(unique_classes & rare_classes)
    rare_bonus = rare_count * 0.15

    score = 0.3 * fg_ratio + 0.4 * class_diversity + 0.3 * rare_bonus
    return score


def generate_skeleton(mask, num_classes=NUM_CLASSES):
    """Generate 1-pixel skeleton from multi-class mask."""
    skeleton = np.zeros_like(mask)
    for cls_id in range(1, num_classes):
        cls_mask = (mask == cls_id).astype(np.uint8)
        if cls_mask.sum() > 0:
            skel = skeletonize(cls_mask > 0).astype(np.uint8)
            skeleton[skel > 0] = cls_id
    return skeleton


def extract_patches_from_image(img_rgb, mask, image_name, max_patches=MAX_PATCHES_PER_IMAGE):
    """Extract top-N patches from a single CAD image + mask pair."""
    h, w = img_rgb.shape[:2]
    candidates = []

    for y in range(0, h - PATCH_SIZE // 2, STRIDE):
        for x in range(0, w - PATCH_SIZE // 2, STRIDE):
            y2 = min(y + PATCH_SIZE, h)
            x2 = min(x + PATCH_SIZE, w)
            y1 = max(0, y2 - PATCH_SIZE)
            x1 = max(0, x2 - PATCH_SIZE)

            mask_patch = mask[y1:y2, x1:x2]
            fg_ratio = (mask_patch > 0).sum() / mask_patch.size

            if fg_ratio < MIN_FG_RATIO:
                continue

            score = compute_patch_score(mask_patch)
            candidates.append({
                "y1": y1, "x1": x1, "y2": y2, "x2": x2,
                "score": score,
                "fg_ratio": fg_ratio,
                "classes": [int(c) for c in set(np.unique(mask_patch)) - {0}],
            })

    if not candidates:
        return []

    # Sort by score, keep top N
    candidates.sort(key=lambda c: c["score"], reverse=True)
    selected = candidates[:max_patches]

    patches = []
    for i, cand in enumerate(selected):
        y1, x1, y2, x2 = cand["y1"], cand["x1"], cand["y2"], cand["x2"]
        img_patch = img_rgb[y1:y2, x1:x2]
        mask_patch = mask[y1:y2, x1:x2]

        # Pad if needed
        ph, pw = img_patch.shape[:2]
        if ph < PATCH_SIZE or pw < PATCH_SIZE:
            img_pad = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
            mask_pad = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
            img_pad[:ph, :pw] = img_patch
            mask_pad[:ph, :pw] = mask_patch
            img_patch = img_pad
            mask_patch = mask_pad

        patches.append({
            "image": img_patch,
            "mask": mask_patch,
            "name": f"cad_{image_name}_p{i:02d}",
            "score": cand["score"],
            "fg_ratio": cand["fg_ratio"],
            "classes": cand["classes"],
        })

    return patches


def run_integration(preview_only=False):
    """Main integration pipeline."""
    logger.info("=" * 60)
    logger.info("CAD ANNOTATION INTEGRATION PIPELINE")
    logger.info("=" * 60)

    # Load class map
    with open(CLASSES_JSON) as f:
        class_map = json.load(f)
    logger.info(f"Classes: {list(class_map.keys())}")

    # Get CAD images that have annotations
    cad_report = pd.read_csv(CAD_REPORT)
    cad_images = []
    for _, row in cad_report.iterrows():
        img_name = row["image"]
        stem = Path(img_name).stem
        mask_path = CAD_MASK_DIR / f"{stem}.png"
        if mask_path.exists():
            img_path = CAD_IMG_DIR / img_name
            if img_path.exists():
                cad_images.append({"img": img_path, "mask": mask_path, "stem": stem})

    logger.info(f"Found {len(cad_images)} CAD images with masks")

    if preview_only:
        logger.info("Preview mode — showing stats only, no patching")
        return

    # ── Extract CAD patches ──
    logger.info(f"\nExtracting patches (size={PATCH_SIZE}, stride={STRIDE})...")
    all_patches = []
    for item in tqdm(cad_images, desc="Extracting CAD patches"):
        img_bgr = safe_imread(item["img"])
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask = np.array(Image.open(item["mask"]))

        patches = extract_patches_from_image(img_rgb, mask, item["stem"])
        all_patches.extend(patches)

    logger.info(f"Extracted {len(all_patches)} CAD patches from {len(cad_images)} images")

    # ── Split CAD patches into train/val ──
    np.random.seed(SEED)
    indices = np.random.permutation(len(all_patches))
    n_train = int(len(all_patches) * TRAIN_RATIO)
    train_idx = set(indices[:n_train])

    # ── Setup combined output directory ──
    (COMBINED_DIR / "images").mkdir(parents=True, exist_ok=True)
    (COMBINED_DIR / "masks").mkdir(parents=True, exist_ok=True)
    (COMBINED_DIR / "centerlines").mkdir(parents=True, exist_ok=True)
    (COMBINED_DIR / "metadata").mkdir(parents=True, exist_ok=True)

    # ── Step 1: Copy existing synthetic patches ──
    logger.info("\nCopying existing synthetic patches...")
    synth_meta = pd.read_csv(SYNTH_PATCH_DIR / "metadata" / "patch_metadata.csv")
    logger.info(f"  Synthetic patches: {len(synth_meta)}")

    # Copy files
    for _, row in tqdm(synth_meta.iterrows(), total=len(synth_meta), desc="Copying synthetic"):
        name = row["patch_name"]

        src_img = SYNTH_PATCH_DIR / "images" / f"{name}.jpg"
        src_mask = SYNTH_PATCH_DIR / "masks" / f"{name}.png"
        src_skel = SYNTH_PATCH_DIR / "centerlines" / f"{name}.png"

        dst_img = COMBINED_DIR / "images" / f"{name}.jpg"
        dst_mask = COMBINED_DIR / "masks" / f"{name}.png"
        dst_skel = COMBINED_DIR / "centerlines" / f"{name}.png"

        if src_img.exists() and not dst_img.exists():
            shutil.copy2(src_img, dst_img)
        if src_mask.exists() and not dst_mask.exists():
            shutil.copy2(src_mask, dst_mask)
        if src_skel.exists() and not dst_skel.exists():
            shutil.copy2(src_skel, dst_skel)

    # ── Step 2: Save CAD patches ──
    logger.info(f"\nSaving {len(all_patches)} CAD patches...")
    cad_meta_rows = []

    for i, patch in enumerate(tqdm(all_patches, desc="Saving CAD patches")):
        name = patch["name"]

        # Save image as JPEG
        img_bgr = cv2.cvtColor(patch["image"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(COMBINED_DIR / "images" / f"{name}.jpg"), img_bgr,
                     [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Save mask as PNG
        Image.fromarray(patch["mask"]).save(str(COMBINED_DIR / "masks" / f"{name}.png"))

        # Generate and save skeleton
        skeleton = generate_skeleton(patch["mask"])
        Image.fromarray(skeleton).save(str(COMBINED_DIR / "centerlines" / f"{name}.png"))

        # Determine split
        split = "train" if i in train_idx else "val"

        # CAD patches get higher sampling weight (1.5-2.5) because:
        # - They are real-world data (more valuable than synthetic)
        # - They are fewer in number (need oversampling to balance)
        base_weight = 1.5  # Real data baseline bonus
        class_bonus = len(patch["classes"]) * 0.15  # More classes = more valuable
        rare_classes = {1, 5, 6, 8}
        rare_bonus = len(set(patch["classes"]) & rare_classes) * 0.2
        sampling_weight = min(3.0, base_weight + class_bonus + rare_bonus)

        cad_meta_rows.append({
            "patch_name": name,
            "source_image": name.replace("cad_", "").rsplit("_p", 1)[0],
            "source_class": "CAD_drawing",
            "split": split,
            "fg_ratio": round(patch["fg_ratio"], 4),
            "classes_present": json.dumps(patch["classes"]),
            "patch_score": round(patch["score"], 4),
            "sampling_weight": round(sampling_weight, 2),
        })

    # ── Step 3: Create combined metadata ──
    logger.info("\nCreating combined metadata...")

    # Add 'source_type' column to distinguish synthetic vs CAD
    synth_meta_copy = synth_meta.copy()
    synth_meta_copy["source_type"] = "synthetic"

    cad_df = pd.DataFrame(cad_meta_rows)
    cad_df["source_type"] = "cad"

    # Align columns
    common_cols = ["patch_name", "source_image", "split", "fg_ratio",
                   "classes_present", "sampling_weight", "source_type"]

    synth_cols = {c: c for c in common_cols if c in synth_meta_copy.columns}
    missing_synth = [c for c in common_cols if c not in synth_meta_copy.columns]
    for c in missing_synth:
        if c == "source_type":
            synth_meta_copy[c] = "synthetic"

    cad_cols = {c: c for c in common_cols if c in cad_df.columns}

    # Combine
    combined_meta = pd.concat([
        synth_meta_copy[common_cols] if all(c in synth_meta_copy.columns for c in common_cols)
        else synth_meta_copy,
        cad_df,
    ], ignore_index=True)

    combined_meta.to_csv(COMBINED_DIR / "metadata" / "patch_metadata.csv", index=False)

    # ── Report ──
    n_synth_train = len(synth_meta[synth_meta["split"] == "train"])
    n_synth_val = len(synth_meta[synth_meta["split"] == "val"])
    n_cad_train = len(cad_df[cad_df["split"] == "train"])
    n_cad_val = len(cad_df[cad_df["split"] == "val"])

    logger.info(f"\n{'=' * 60}")
    logger.info("INTEGRATION COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"\nCombined dataset: {COMBINED_DIR}")
    logger.info(f"\n  {'Source':<20} {'Train':>8} {'Val':>8} {'Total':>8}")
    logger.info(f"  {'-'*46}")
    logger.info(f"  {'Synthetic':<20} {n_synth_train:>8} {n_synth_val:>8} {n_synth_train+n_synth_val:>8}")
    logger.info(f"  {'CAD (auto-annot)':<20} {n_cad_train:>8} {n_cad_val:>8} {n_cad_train+n_cad_val:>8}")
    logger.info(f"  {'-'*46}")
    logger.info(f"  {'TOTAL':<20} {n_synth_train+n_cad_train:>8} {n_synth_val+n_cad_val:>8} "
                f"{len(combined_meta):>8}")

    # Check disk usage
    total_bytes = sum(f.stat().st_size for f in COMBINED_DIR.rglob("*") if f.is_file())
    logger.info(f"\n  Disk used: {total_bytes / 1e9:.2f} GB")
    logger.info(f"\n  To train on combined data, use:")
    logger.info(f'    python trainer.py --patch_root "{COMBINED_DIR}"')


def main():
    parser = argparse.ArgumentParser(description="Integrate CAD annotations into training")
    parser.add_argument("--preview_only", action="store_true",
                        help="Show stats without creating patches")
    args = parser.parse_args()
    run_integration(preview_only=args.preview_only)


if __name__ == "__main__":
    main()
