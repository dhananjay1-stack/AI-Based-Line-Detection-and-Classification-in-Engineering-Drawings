#!/usr/bin/env python3
"""
Engineering Drawing Dataset: Audit, Clean, Normalize & Split Pipeline
=====================================================================

Usage:
    python dataset_pipeline.py audit    --dataset_root <path> --out_dir <path>
    python dataset_pipeline.py clean    --dataset_root <path> --out_dir <path>
    python dataset_pipeline.py split    --out_dir <path>
    python dataset_pipeline.py preview  --out_dir <path>
    python dataset_pipeline.py run_all  --dataset_root <path> --out_dir <path>

Requirements:
    pip install numpy opencv-python Pillow tqdm pandas
"""

import os
import sys
import json
import argparse
import logging
import random
import hashlib
import csv
from pathlib import Path
from collections import defaultdict, OrderedDict
from datetime import datetime

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import pandas as pd

# =============================================================================
# Logging Setup
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dataset_pipeline")

# =============================================================================
# Constants
# =============================================================================
SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}

# Classes to REMOVE from the dataset
CLASSES_TO_REMOVE = {"Arrowhead", "Dimension_text", "Section_hatching_(cross-hatch)"}

# Canonical class ordering (alphabetical, after removal)
# Background = 0, then 1..9
KEPT_CLASSES = [
    "Center_line",
    "Dimension_lines",
    "Extension_line",
    "Feature_Visible",
    "Leader_line",
    "Phantom_lines",
    "break_line",
    "cutting_plane",
    "hidden_lines",
]

# Map from the raw directory names (may contain spaces/parens) to canonical names
DIR_NAME_TO_CANONICAL = {
    "Center_line": "Center_line",
    "Dimension_lines": "Dimension_lines",
    "Extension_line": "Extension_line",
    "Feature_Visible": "Feature_Visible",
    "Leader_line": "Leader_line",
    "Phantom_lines": "Phantom_lines",
    "Section hatching (cross-hatch)": "Section_hatching_(cross-hatch)",
    "break_line": "break_line",
    "cutting_plane": "cutting_plane",
    "hidden_lines": "hidden_lines",
    "Arrowhead": "Arrowhead",
    "Dimension_text": "Dimension_text",
}

# Color palette for visualization (10 classes + background)
VIS_COLORS = np.array([
    [0,   0,   0],    # 0: Background (black)
    [255, 0,   0],    # 1: Center_line (red)
    [0,   255, 0],    # 2: Dimension_lines (green)
    [0,   0,   255],  # 3: Extension_line (blue)
    [255, 255, 0],    # 4: Feature_Visible (yellow)
    [255, 0,   255],  # 5: Leader_line (magenta)
    [0,   255, 255],  # 6: Phantom_lines (cyan)
    [255, 128, 0],    # 7: Section_hatching (orange)
    [128, 0,   255],  # 8: break_line (purple)
    [0,   128, 128],  # 9: cutting_plane (teal)
    [128, 128, 0],    # 10: hidden_lines (olive)
], dtype=np.uint8)


# =============================================================================
# Helper Functions
# =============================================================================

def set_seed(seed=SEED):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def get_file_hash(filepath, chunk_size=8192):
    """Compute MD5 hash of a file for duplicate detection."""
    h = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(chunk_size):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def safe_imread(path, flags=cv2.IMREAD_UNCHANGED):
    """Read an image safely, handling unicode paths on Windows."""
    try:
        # cv2.imread can fail with unicode paths on Windows
        arr = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(arr, flags)
        return img
    except Exception:
        return None


def safe_imwrite(path, img):
    """Write an image safely, handling unicode paths on Windows."""
    try:
        ext = Path(path).suffix
        success, buf = cv2.imencode(ext, img)
        if success:
            buf.tofile(str(path))
            return True
    except Exception:
        pass
    return False


def scan_essential_dataset(dataset_root):
    """
    Scan the Essential/ folder structure.
    
    Returns:
        class_dirs: dict {canonical_class_name: raw_dir_name}
        samples: list of dicts with keys:
            'stem', 'class_raw', 'class_canonical', 'image_path', 'mask_path',
            'image_exists', 'mask_exists'
    """
    essential_root = Path(dataset_root) / "Essential"
    if not essential_root.exists():
        logger.error(f"Essential directory not found: {essential_root}")
        return {}, []

    class_dirs = {}
    samples = []

    for class_dir in sorted(essential_root.iterdir()):
        if not class_dir.is_dir():
            continue

        raw_name = class_dir.name
        canonical = DIR_NAME_TO_CANONICAL.get(raw_name, raw_name.replace(" ", "_"))
        class_dirs[canonical] = raw_name

        img_dir = class_dir / "images"
        mask_dir = class_dir / "masks"

        # Collect all image stems
        img_stems = {}
        if img_dir.exists():
            for f in img_dir.iterdir():
                if f.suffix.lower() in IMG_EXTS:
                    img_stems[f.stem] = f

        # Collect all mask stems
        mask_stems = {}
        if mask_dir.exists():
            for f in mask_dir.iterdir():
                if f.suffix.lower() in IMG_EXTS:
                    mask_stems[f.stem] = f

        # Union of all stems
        all_stems = set(img_stems.keys()) | set(mask_stems.keys())

        for stem in sorted(all_stems):
            samples.append({
                "stem": stem,
                "class_raw": raw_name,
                "class_canonical": canonical,
                "image_path": str(img_stems[stem]) if stem in img_stems else None,
                "mask_path": str(mask_stems[stem]) if stem in mask_stems else None,
                "image_exists": stem in img_stems,
                "mask_exists": stem in mask_stems,
            })

    return class_dirs, samples


def scan_engineering_drawings(eng_dir):
    """Catalog the unannotated engineering drawings."""
    eng_path = Path(eng_dir)
    if not eng_path.exists():
        return []

    catalog = []
    for f in sorted(eng_path.iterdir()):
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            try:
                img = Image.open(f)
                w, h = img.size
                mode = img.mode
                img.close()
                catalog.append({
                    "filename": f.name,
                    "format": f.suffix.lower(),
                    "width": w,
                    "height": h,
                    "mode": mode,
                    "size_bytes": f.stat().st_size,
                })
            except Exception as e:
                catalog.append({
                    "filename": f.name,
                    "format": f.suffix.lower(),
                    "width": None,
                    "height": None,
                    "mode": None,
                    "size_bytes": f.stat().st_size,
                    "error": str(e),
                })
    return catalog


# =============================================================================
# Phase 1: AUDIT
# =============================================================================

def run_audit(args):
    """
    Full audit of the dataset:
    - Image-mask pairing
    - Corrupted/unreadable files
    - Mask pixel value analysis
    - Class frequency statistics
    - Engineering drawing catalog
    """
    logger.info("=" * 70)
    logger.info("PHASE 1: DATASET AUDIT")
    logger.info("=" * 70)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Scan Essential dataset ---
    logger.info(f"Scanning dataset root: {args.dataset_root}")
    class_dirs, samples = scan_essential_dataset(args.dataset_root)

    total_samples = len(samples)
    logger.info(f"Found {len(class_dirs)} classes, {total_samples} total entries")

    # --- 2. Check pairing issues ---
    missing_images = [s for s in samples if not s["image_exists"]]
    missing_masks = [s for s in samples if not s["mask_exists"]]
    paired = [s for s in samples if s["image_exists"] and s["mask_exists"]]

    logger.info(f"  Paired (image+mask): {len(paired)}")
    logger.info(f"  Missing images:      {len(missing_images)}")
    logger.info(f"  Missing masks:       {len(missing_masks)}")

    # --- 3. Check for corrupted files and analyze masks ---
    logger.info("Analyzing mask encoding and checking for corruption...")
    
    class_stats = defaultdict(lambda: {
        "total": 0,
        "paired": 0,
        "missing_image": 0,
        "missing_mask": 0,
        "corrupted_image": 0,
        "corrupted_mask": 0,
        "mask_binary_clean": 0,        # Only 0 and 255
        "mask_has_antialiasing": 0,     # Intermediate values
        "mask_empty": 0,               # All zeros
        "total_mask_pixels": 0,
        "total_foreground_pixels": 0,
        "image_sizes": set(),
        "mask_unique_values": set(),
    })

    corrupted_files = []
    problematic_samples = []

    for sample in tqdm(paired, desc="Auditing paired samples"):
        cls = sample["class_canonical"]
        stats = class_stats[cls]
        stats["total"] += 1
        stats["paired"] += 1

        # Check image
        img = safe_imread(sample["image_path"])
        if img is None:
            stats["corrupted_image"] += 1
            corrupted_files.append({
                "path": sample["image_path"],
                "type": "image",
                "class": cls,
                "issue": "unreadable",
            })
            continue

        h, w = img.shape[:2]
        stats["image_sizes"].add(f"{w}x{h}")

        # Check mask
        mask = safe_imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            stats["corrupted_mask"] += 1
            corrupted_files.append({
                "path": sample["mask_path"],
                "type": "mask",
                "class": cls,
                "issue": "unreadable",
            })
            continue

        # Mask analysis
        unique_vals = np.unique(mask)
        stats["mask_unique_values"].update(unique_vals.tolist())

        total_px = mask.size
        fg_px = np.count_nonzero(mask)
        stats["total_mask_pixels"] += total_px
        stats["total_foreground_pixels"] += fg_px

        if fg_px == 0:
            stats["mask_empty"] += 1
            problematic_samples.append({
                "stem": sample["stem"],
                "class": cls,
                "issue": "empty_mask",
                "mask_path": sample["mask_path"],
            })

        # Check if mask is clean binary (only 0 and 255)
        if set(unique_vals.tolist()).issubset({0, 255}):
            stats["mask_binary_clean"] += 1
        else:
            stats["mask_has_antialiasing"] += 1

        # Size mismatch check
        mh, mw = mask.shape[:2]
        if (mh, mw) != (h, w):
            problematic_samples.append({
                "stem": sample["stem"],
                "class": cls,
                "issue": f"size_mismatch: image={w}x{h}, mask={mw}x{mh}",
                "mask_path": sample["mask_path"],
            })

    # --- 4. Process missing files ---
    for sample in missing_images:
        cls = sample["class_canonical"]
        class_stats[cls]["total"] += 1
        class_stats[cls]["missing_image"] += 1

    for sample in missing_masks:
        cls = sample["class_canonical"]
        class_stats[cls]["total"] += 1
        class_stats[cls]["missing_mask"] += 1

    # --- 5. Scan engineering drawings ---
    eng_dir = Path(args.eng_drawings_dir)
    eng_catalog = scan_engineering_drawings(eng_dir)
    logger.info(f"Engineering drawings cataloged: {len(eng_catalog)}")

    # --- 6. Build report ---
    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset_root": str(args.dataset_root),
        "summary": {
            "total_classes": len(class_dirs),
            "total_entries": total_samples,
            "total_paired": len(paired),
            "total_missing_images": len(missing_images),
            "total_missing_masks": len(missing_masks),
            "total_corrupted": len(corrupted_files),
            "total_problematic": len(problematic_samples),
            "engineering_drawings_count": len(eng_catalog),
            "engineering_drawings_annotated": 0,
        },
        "classes_to_remove": list(CLASSES_TO_REMOVE),
        "classes_to_keep": KEPT_CLASSES,
        "class_details": {},
        "corrupted_files": corrupted_files,
        "problematic_samples": problematic_samples[:100],  # Cap at 100
        "engineering_drawings_summary": {
            "total": len(eng_catalog),
            "formats": dict(pd.DataFrame(eng_catalog)["format"].value_counts()) if eng_catalog else {},
            "has_masks": False,
        },
    }

    # Class details
    for cls in sorted(class_stats.keys()):
        s = class_stats[cls]
        fg_ratio = s["total_foreground_pixels"] / max(s["total_mask_pixels"], 1)
        report["class_details"][cls] = {
            "total_entries": s["total"],
            "paired": s["paired"],
            "missing_image": s["missing_image"],
            "missing_mask": s["missing_mask"],
            "corrupted_image": s["corrupted_image"],
            "corrupted_mask": s["corrupted_mask"],
            "mask_binary_clean": s["mask_binary_clean"],
            "mask_has_antialiasing": s["mask_has_antialiasing"],
            "mask_empty": s["mask_empty"],
            "foreground_ratio": round(fg_ratio, 6),
            "image_sizes": sorted(list(s["image_sizes"])),
            "will_be_removed": cls in CLASSES_TO_REMOVE,
        }

    # Save JSON report
    report_json_path = out_dir / "audit_report.json"
    with open(report_json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Audit report (JSON): {report_json_path}")

    # Save CSV report
    csv_rows = []
    for cls, details in report["class_details"].items():
        row = {"class": cls}
        row.update(details)
        row["image_sizes"] = "; ".join(details["image_sizes"])
        csv_rows.append(row)

    report_csv_path = out_dir / "audit_report.csv"
    pd.DataFrame(csv_rows).to_csv(report_csv_path, index=False)
    logger.info(f"Audit report (CSV): {report_csv_path}")

    # Save engineering drawings catalog
    eng_catalog_path = out_dir / "engineering_drawings"
    eng_catalog_path.mkdir(exist_ok=True)
    with open(eng_catalog_path / "catalog.json", "w") as f:
        json.dump(eng_catalog, f, indent=2)
    logger.info(f"Engineering drawings catalog: {eng_catalog_path / 'catalog.json'}")

    # --- 7. Print summary ---
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    print(f"{'Class':<35} {'Paired':>7} {'Empty':>6} {'AA':>6} {'FG%':>8} {'Remove':>7}")
    print("-" * 70)
    for cls in sorted(class_stats.keys()):
        s = class_stats[cls]
        fg = s["total_foreground_pixels"] / max(s["total_mask_pixels"], 1) * 100
        removed = "YES" if cls in CLASSES_TO_REMOVE else ""
        aa = s["mask_has_antialiasing"]
        print(f"{cls:<35} {s['paired']:>7} {s['mask_empty']:>6} {aa:>6} {fg:>7.3f}% {removed:>7}")
    print("-" * 70)
    print(f"Total paired: {len(paired)}")
    print(f"Corrupted files: {len(corrupted_files)}")
    print(f"Problematic samples: {len(problematic_samples)}")
    print(f"Engineering drawings (unannotated): {len(eng_catalog)}")
    print("=" * 70)

    return report


# =============================================================================
# Phase 2: CLEAN
# =============================================================================

def run_clean(args):
    """
    Clean and normalize the dataset:
    - Remove unwanted classes (Arrowhead, Dimension_text)
    - Binarize masks (threshold → 0/1)
    - Normalize images to PNG, RGB
    - Build merged multi-class masks
    - Create new class mapping
    """
    logger.info("=" * 70)
    logger.info("PHASE 2: CLEAN & NORMALIZE")
    logger.info("=" * 70)

    out_dir = Path(args.out_dir)
    cleaned_dir = out_dir / "cleaned"
    img_out = cleaned_dir / "images"
    mask_out = cleaned_dir / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    # --- 1. Build new class mapping ---
    new_class_map = OrderedDict()
    for idx, cls_name in enumerate(KEPT_CLASSES, start=1):
        new_class_map[cls_name] = idx

    classes_path = out_dir / "classes_cleaned.json"
    with open(classes_path, "w") as f:
        json.dump(new_class_map, f, indent=2)
    logger.info(f"New class mapping ({len(new_class_map)} classes): {classes_path}")
    for name, idx in new_class_map.items():
        logger.info(f"  {idx:>2}: {name}")

    # --- 2. Scan dataset ---
    class_dirs, samples = scan_essential_dataset(args.dataset_root)

    # Filter to only paired, kept classes
    kept_samples = [
        s for s in samples
        if s["image_exists"] and s["mask_exists"]
        and s["class_canonical"] not in CLASSES_TO_REMOVE
    ]

    removed_samples = [
        s for s in samples
        if s["class_canonical"] in CLASSES_TO_REMOVE
    ]

    logger.info(f"Kept samples: {len(kept_samples)}")
    logger.info(f"Removed samples (Arrowhead + Dimension_text): {len(removed_samples)}")

    # --- 3. Group samples by stem to find unique images ---
    # Each image may appear in multiple class folders (same image, different class mask)
    # For merged masks, we group by stem
    stem_groups = defaultdict(list)
    for s in kept_samples:
        stem_groups[s["stem"]].append(s)

    logger.info(f"Unique image stems: {len(stem_groups)}")

    # --- 4. Process each unique image ---
    stats_before = defaultdict(int)
    stats_after = defaultdict(int)
    empty_masks = []
    processed = 0
    skipped = 0

    for stem, group in tqdm(stem_groups.items(), desc="Cleaning & merging"):
        # Use the first available image as the source
        img_path = group[0]["image_path"]
        img = safe_imread(img_path)
        if img is None:
            skipped += 1
            continue

        # Ensure RGB (3 channels)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        h, w = img.shape[:2]

        # Create merged multi-class mask canvas
        merged_mask = np.zeros((h, w), dtype=np.uint8)

        classes_present = []

        for sample in group:
            cls = sample["class_canonical"]
            cls_id = new_class_map.get(cls)
            if cls_id is None:
                continue

            # Read the binary mask
            raw_mask = safe_imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
            if raw_mask is None:
                continue

            # Resize mask if needed
            mh, mw = raw_mask.shape[:2]
            if (mh, mw) != (h, w):
                raw_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Binarize: threshold at 127
            _, binary = cv2.threshold(raw_mask, 127, 1, cv2.THRESH_BINARY)

            # Count pixels before/after
            stats_before[cls] += np.count_nonzero(raw_mask)
            stats_after[cls] += np.count_nonzero(binary)

            # Paint onto merged mask (later classes overwrite earlier ones)
            # This is fine since each image typically has only ONE class mask
            merged_mask[binary > 0] = cls_id
            classes_present.append(cls)

        # Check if merged mask is empty
        if np.count_nonzero(merged_mask) == 0:
            empty_masks.append(stem)
            skipped += 1
            continue

        # Verify mask values are valid
        unique_vals = np.unique(merged_mask)
        assert all(v <= len(KEPT_CLASSES) for v in unique_vals), \
            f"Invalid mask values for {stem}: {unique_vals}"

        # Save normalized image as PNG
        out_img_path = img_out / f"{stem}.png"
        safe_imwrite(str(out_img_path), img)

        # Save merged mask as single-channel PNG (pixel = class_id)
        out_mask_path = mask_out / f"{stem}.png"
        # Use PIL for lossless save with no compression artifacts
        Image.fromarray(merged_mask).save(str(out_mask_path))

        processed += 1

    logger.info(f"Processed: {processed}, Skipped: {skipped}, Empty masks: {len(empty_masks)}")

    # --- 5. Save class distribution before/after ---
    dist_rows = []
    for cls in KEPT_CLASSES:
        cls_id = new_class_map[cls]
        count_before = stats_before.get(cls, 0)
        count_after = stats_after.get(cls, 0)
        # Count samples per class
        cls_samples = sum(1 for s in kept_samples if s["class_canonical"] == cls)
        dist_rows.append({
            "class_id": cls_id,
            "class_name": cls,
            "sample_count": cls_samples,
            "total_fg_pixels_before_binarize": count_before,
            "total_fg_pixels_after_binarize": count_after,
        })

    # Also add removed classes
    for cls in CLASSES_TO_REMOVE:
        cls_samples = sum(1 for s in samples if s["class_canonical"] == cls)
        dist_rows.append({
            "class_id": "REMOVED",
            "class_name": cls,
            "sample_count": cls_samples,
            "total_fg_pixels_before_binarize": "N/A",
            "total_fg_pixels_after_binarize": 0,
        })

    dist_path = out_dir / "class_distribution.csv"
    pd.DataFrame(dist_rows).to_csv(dist_path, index=False)
    logger.info(f"Class distribution: {dist_path}")

    # Save empty mask list
    if empty_masks:
        empty_path = out_dir / "empty_masks.txt"
        with open(empty_path, "w") as f:
            f.write("\n".join(empty_masks))
        logger.info(f"Empty masks list ({len(empty_masks)}): {empty_path}")

    # --- 6. Print summary ---
    print("\n" + "=" * 70)
    print("CLEANING SUMMARY")
    print("=" * 70)
    print(f"{'Class':<35} {'ID':>3} {'Samples':>8} {'FG Pixels (after)':>18}")
    print("-" * 70)
    for row in dist_rows:
        if row["class_id"] != "REMOVED":
            print(f"{row['class_name']:<35} {row['class_id']:>3} {row['sample_count']:>8} {row['total_fg_pixels_after_binarize']:>18}")
    print("-" * 70)
    print(f"Total processed images: {processed}")
    print(f"Removed classes: {', '.join(CLASSES_TO_REMOVE)}")
    print(f"Empty masks skipped: {len(empty_masks)}")
    print("=" * 70)


# =============================================================================
# Phase 3: SPLIT
# =============================================================================

def run_split(args):
    """
    Create reproducible train/val/test splits (80/10/10).
    Stratified by primary class.
    """
    logger.info("=" * 70)
    logger.info("PHASE 3: CREATE SPLITS")
    logger.info("=" * 70)

    set_seed(SEED)

    out_dir = Path(args.out_dir)
    cleaned_dir = out_dir / "cleaned"
    img_dir = cleaned_dir / "images"
    mask_dir = cleaned_dir / "masks"

    if not img_dir.exists():
        logger.error(f"Cleaned images not found: {img_dir}. Run 'clean' first.")
        return

    # Load class map
    classes_path = out_dir / "classes_cleaned.json"
    if not classes_path.exists():
        logger.error(f"Class mapping not found: {classes_path}. Run 'clean' first.")
        return

    with open(classes_path, "r") as f:
        class_map = json.load(f)

    id_to_class = {v: k for k, v in class_map.items()}

    # Get all valid samples (image must have matching mask)
    all_images = sorted(img_dir.glob("*.png"))
    valid_samples = []

    for img_path in all_images:
        mask_path = mask_dir / img_path.name
        if mask_path.exists():
            valid_samples.append(img_path.stem)

    logger.info(f"Total valid samples: {len(valid_samples)}")

    # Determine primary class for each sample (the dominant non-background class)
    sample_classes = {}
    for stem in tqdm(valid_samples, desc="Determining primary classes"):
        mask_path = mask_dir / f"{stem}.png"
        mask = np.array(Image.open(mask_path))

        # Find the most common non-zero class
        unique, counts = np.unique(mask, return_counts=True)
        non_bg = [(u, c) for u, c in zip(unique, counts) if u > 0]

        if non_bg:
            primary_cls = max(non_bg, key=lambda x: x[1])[0]
            sample_classes[stem] = id_to_class.get(primary_cls, f"class_{primary_cls}")
        else:
            sample_classes[stem] = "background_only"

    # Group by class for stratified splitting
    class_groups = defaultdict(list)
    for stem, cls in sample_classes.items():
        class_groups[cls].append(stem)

    train_stems = []
    val_stems = []
    test_stems = []

    for cls, stems in sorted(class_groups.items()):
        random.shuffle(stems)
        n = len(stems)
        n_test = max(1, int(n * 0.1))
        n_val = max(1, int(n * 0.1))
        n_train = n - n_val - n_test

        test_stems.extend(stems[:n_test])
        val_stems.extend(stems[n_test:n_test + n_val])
        train_stems.extend(stems[n_test + n_val:])

    # Shuffle each split
    random.shuffle(train_stems)
    random.shuffle(val_stems)
    random.shuffle(test_stems)

    # Verify no overlap
    train_set = set(train_stems)
    val_set = set(val_stems)
    test_set = set(test_stems)
    assert train_set.isdisjoint(val_set), "Train/Val overlap detected!"
    assert train_set.isdisjoint(test_set), "Train/Test overlap detected!"
    assert val_set.isdisjoint(test_set), "Val/Test overlap detected!"

    total = len(train_stems) + len(val_stems) + len(test_stems)
    logger.info(f"Split: Train={len(train_stems)} ({len(train_stems)/total*100:.1f}%), "
                f"Val={len(val_stems)} ({len(val_stems)/total*100:.1f}%), "
                f"Test={len(test_stems)} ({len(test_stems)/total*100:.1f}%)")

    # Save splits
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    for name, stems in [("train", train_stems), ("val", val_stems), ("test", test_stems)]:
        split_path = splits_dir / f"{name}.txt"
        with open(split_path, "w") as f:
            for stem in stems:
                f.write(f"{stem}\n")
        logger.info(f"  {name}: {len(stems)} samples → {split_path}")

    # Save split statistics per class
    split_stats = []
    for cls in sorted(class_groups.keys()):
        stems = class_groups[cls]
        n_train = sum(1 for s in stems if s in train_set)
        n_val = sum(1 for s in stems if s in val_set)
        n_test = sum(1 for s in stems if s in test_set)
        split_stats.append({
            "class": cls,
            "total": len(stems),
            "train": n_train,
            "val": n_val,
            "test": n_test,
        })

    split_stats_path = out_dir / "split_statistics.csv"
    pd.DataFrame(split_stats).to_csv(split_stats_path, index=False)
    logger.info(f"Split statistics: {split_stats_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)
    print(f"{'Class':<35} {'Total':>6} {'Train':>6} {'Val':>6} {'Test':>6}")
    print("-" * 70)
    for row in split_stats:
        print(f"{row['class']:<35} {row['total']:>6} {row['train']:>6} {row['val']:>6} {row['test']:>6}")
    print("-" * 70)
    print(f"{'TOTAL':<35} {total:>6} {len(train_stems):>6} {len(val_stems):>6} {len(test_stems):>6}")
    print("=" * 70)


# =============================================================================
# Phase 4: PREVIEW
# =============================================================================

def run_preview(args):
    """
    Generate visual QA samples: side-by-side image + colored mask overlay.
    """
    logger.info("=" * 70)
    logger.info("PHASE 4: GENERATE PREVIEWS")
    logger.info("=" * 70)

    set_seed(SEED)

    out_dir = Path(args.out_dir)
    cleaned_dir = out_dir / "cleaned"
    img_dir = cleaned_dir / "images"
    mask_dir = cleaned_dir / "masks"
    splits_dir = out_dir / "splits"
    preview_dir = out_dir / "preview"
    preview_dir.mkdir(exist_ok=True)

    # Load class map for legend
    classes_path = out_dir / "classes_cleaned.json"
    if classes_path.exists():
        with open(classes_path, "r") as f:
            class_map = json.load(f)
        id_to_class = {v: k for k, v in class_map.items()}
    else:
        id_to_class = {}

    n_preview = args.preview_n

    for split_name in ["train", "val", "test"]:
        split_file = splits_dir / f"{split_name}.txt"
        if not split_file.exists():
            logger.warning(f"Split file not found: {split_file}")
            continue

        with open(split_file, "r") as f:
            stems = [line.strip() for line in f if line.strip()]

        # Sample random stems
        sample_stems = random.sample(stems, min(n_preview, len(stems)))

        for i, stem in enumerate(sample_stems):
            img_path = img_dir / f"{stem}.png"
            mask_path = mask_dir / f"{stem}.png"

            if not img_path.exists() or not mask_path.exists():
                continue

            img = safe_imread(str(img_path))
            mask = np.array(Image.open(mask_path))

            if img is None:
                continue

            # Ensure 3-channel for overlay
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            h, w = img.shape[:2]

            # Create colored mask overlay
            color_mask = np.zeros_like(img)
            unique_classes = np.unique(mask)
            for cls_id in unique_classes:
                if cls_id == 0:
                    continue
                if cls_id < len(VIS_COLORS):
                    color = VIS_COLORS[cls_id]
                else:
                    color = np.array([128, 128, 128], dtype=np.uint8)
                color_mask[mask == cls_id] = color[::-1]  # RGB to BGR

            # Blend overlay
            overlay = cv2.addWeighted(img, 0.6, color_mask, 0.4, 0)

            # Create legend
            legend_height = 30 * (len(unique_classes))
            legend_width = 300
            legend = np.ones((max(legend_height, h), legend_width, 3), dtype=np.uint8) * 40

            y_pos = 20
            for cls_id in sorted(unique_classes):
                if cls_id == 0:
                    continue
                cls_name = id_to_class.get(cls_id, f"class_{cls_id}")
                if cls_id < len(VIS_COLORS):
                    color = VIS_COLORS[cls_id][::-1]  # RGB to BGR
                else:
                    color = (128, 128, 128)

                # Draw color swatch
                cv2.rectangle(legend, (10, y_pos - 10), (30, y_pos + 10), color.tolist(), -1)
                # Draw text
                cv2.putText(legend, f"{cls_id}: {cls_name}", (40, y_pos + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                y_pos += 30

            # Resize legend to match image height
            legend = cv2.resize(legend, (legend_width, h), interpolation=cv2.INTER_NEAREST)

            # Concatenate: image | overlay | legend
            combined = np.hstack([img, overlay, legend])

            # Add title
            title_bar = np.ones((40, combined.shape[1], 3), dtype=np.uint8) * 30
            cv2.putText(title_bar, f"[{split_name.upper()}] {stem}",
                       (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            combined = np.vstack([title_bar, combined])

            # Save
            out_path = preview_dir / f"{split_name}_sample_{i:03d}.png"
            safe_imwrite(str(out_path), combined)

        logger.info(f"  {split_name}: {min(n_preview, len(stems))} previews → {preview_dir}")

    logger.info(f"Preview images saved to: {preview_dir}")


# =============================================================================
# Phase 5: RUN ALL
# =============================================================================

def run_all(args):
    """Run all phases sequentially."""
    logger.info("Running complete pipeline: audit → clean → split → preview")
    run_audit(args)
    run_clean(args)
    run_split(args)
    run_preview(args)
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)


# =============================================================================
# CLI
# =============================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="Engineering Drawing Dataset: Audit, Clean, Normalize & Split Pipeline"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Pipeline phase to run")

    # Shared args
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--dataset_root", type=str,
        default=r"C:\Users\Admin\line_detection\output\dataset_cleaned",
        help="Root of Essential dataset (contains Essential/ subfolder)"
    )
    parent.add_argument(
        "--eng_drawings_dir", type=str,
        default=r"C:\Users\Admin\line_detection\Approach 2\Engineering_Drawings",
        help="Path to engineering drawings folder"
    )
    parent.add_argument(
        "--out_dir", type=str,
        default=r"C:\Users\Admin\line_detection\Approach 2\pipeline_output",
        help="Output directory for all pipeline artifacts"
    )
    parent.add_argument("--verbose", action="store_true", help="Enable debug logging")

    # Subcommands
    subparsers.add_parser("audit", parents=[parent], help="Phase 1: Audit dataset")
    subparsers.add_parser("clean", parents=[parent], help="Phase 2: Clean & normalize")

    p_split = subparsers.add_parser("split", parents=[parent], help="Phase 3: Create splits")

    p_preview = subparsers.add_parser("preview", parents=[parent], help="Phase 4: Generate previews")
    p_preview.add_argument("--preview_n", type=int, default=10, help="Previews per split")

    p_all = subparsers.add_parser("run_all", parents=[parent], help="Run all phases")
    p_all.add_argument("--preview_n", type=int, default=10, help="Previews per split")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if hasattr(args, "verbose") and args.verbose:
        logger.setLevel(logging.DEBUG)

    mode_map = {
        "audit": run_audit,
        "clean": run_clean,
        "split": run_split,
        "preview": run_preview,
        "run_all": run_all,
    }

    mode_map[args.mode](args)
