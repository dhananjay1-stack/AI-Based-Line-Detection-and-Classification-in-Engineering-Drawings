"""
Prepare Annotation Dataset from CVAT Exports (v2)
==================================================

Processes per-line-type CVAT annotation exports into a unified
multi-class patch dataset ready for fine-tuning.

Handles TWO annotation formats:
  1. Segmentation Mask 1.1  →  color-coded PNGs in SegmentationClass/
  2. CVAT for Images 1.1    →  annotations.xml with polygon/polyline/mask (RLE)

For line types that have ONLY XML (like Leader_Line), the script
rasterizes the vector annotations into pixel-level masks.

Low-count classes are augmented with extra copies to balance the dataset.

Usage:
    python prepare_annotation_dataset.py
    python prepare_annotation_dataset.py --lines Dimension_lines Extension_line Feature_Visible
    python prepare_annotation_dataset.py --patch_size 512 --min_fg_ratio 0.01
"""

import os
import sys
import json
import csv
import random
import logging
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

try:
    import albumentations as A
except ImportError:
    print("Install albumentations: pip install albumentations")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_annotation")

# ─── Master class mapping ────────────────────────────────────────────────────

MASTER_CLASSES = {
    "Background": 0,
    "Center_line": 1,
    "Dimension_lines": 2,
    "Extension_line": 3,
    "Feature_Visible": 4,
    "Leader_line": 5,
    "Phantom_lines": 6,
    "break_line": 7,
    "cutting_plane": 8,
    "hidden_lines": 9,
    "Section_hatching": 10,
}

# Map CVAT label names and folder names to canonical names
NAME_ALIASES = {
    # Dimension
    "Dimension_lines": "Dimension_lines", "Dimension_line": "Dimension_lines",
    "dimension_lines": "Dimension_lines", "dimension_line": "Dimension_lines",
    # Extension
    "Extension_line": "Extension_line", "Extension_lines": "Extension_line",
    "Extention Line": "Extension_line", "Extention_line": "Extension_line",
    "extention line": "Extension_line", "extention_line": "Extension_line",
    "extension_line": "Extension_line",
    # Center
    "Center_line": "Center_line", "Center_Line": "Center_line",
    "center_line": "Center_line", "Centre_line": "Center_line",
    "Center line": "Center_line",
    # Feature Visible
    "Feature_Visible": "Feature_Visible", "feature_visible": "Feature_Visible",
    "Feature_visible": "Feature_Visible", "Feature_lines": "Feature_Visible",
    "feature_lines": "Feature_Visible",
    # Leader
    "Leader_line": "Leader_line", "Leader_Line": "Leader_line",
    "leader_line": "Leader_line", "Leader_lines": "Leader_line",
    # Phantom
    "Phantom_lines": "Phantom_lines", "phantom_lines": "Phantom_lines",
    "Phantom_line": "Phantom_lines",
    # Break
    "break_line": "break_line", "Break_line": "break_line",
    # Cutting plane
    "cutting_plane": "cutting_plane", "Cutting_plane": "cutting_plane",
    "Cutting_Plane": "cutting_plane",
    # Hidden
    "hidden_lines": "hidden_lines", "Hidden_lines": "hidden_lines",
    "hidden_line": "hidden_lines",
    # Section hatching
    "Section hatching (cross-hatch)": "Section_hatching",
    "Section_hatching": "Section_hatching", "section_hatching": "Section_hatching",
    "Section hatching": "Section_hatching", "Section_Hatching": "Section_hatching",
    "Section Hatching": "Section_hatching",
}


def normalize_class_name(name):
    """Convert any variant of a class name to its canonical form."""
    name_stripped = name.strip()
    if name_stripped in NAME_ALIASES:
        return NAME_ALIASES[name_stripped]
    for alias, canonical in NAME_ALIASES.items():
        if alias.lower() == name_stripped.lower():
            return canonical
    return name_stripped.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")


# ─── Discovery ───────────────────────────────────────────────────────────────

def discover_line_types(annotation_dir):
    """Auto-discover line type folders."""
    annotation_dir = Path(annotation_dir)
    line_types = {}
    for item in sorted(annotation_dir.iterdir()):
        if not item.is_dir():
            continue
        if item.name.lower() in ("engineering_drawings", "raw_images", "images"):
            continue
        canonical = normalize_class_name(item.name)
        if canonical in MASTER_CLASSES:
            line_types[canonical] = item
            logger.info(f"  Found: {item.name} → {canonical} (class {MASTER_CLASSES[canonical]})")
        else:
            new_id = max(MASTER_CLASSES.values()) + 1
            MASTER_CLASSES[canonical] = new_id
            line_types[canonical] = item
            logger.info(f"  New class: {item.name} → {canonical} (assigned ID {new_id})")
    return line_types


def find_raw_images_dir(annotation_dir):
    """Find the raw engineering drawings directory."""
    annotation_dir = Path(annotation_dir)
    candidates = [
        annotation_dir / "Engineering_Drawings" / "Engineering_Drawings",
        annotation_dir / "Engineering_Drawings",
        annotation_dir / "raw_images",
        annotation_dir / "images",
    ]
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            imgs = list(cand.glob("*.png")) + list(cand.glob("*.jpg"))
            if imgs:
                return cand
    return None


# ─── Segmentation Mask Parsing ────────────────────────────────────────────────

def find_segmentation_mask_dir(line_folder):
    """Find the SegmentationClass folder inside a CVAT export."""
    line_folder = Path(line_folder)
    for item in line_folder.iterdir():
        if item.is_dir() and "segmentation" in item.name.lower():
            seg_class_dir = item / "SegmentationClass"
            if seg_class_dir.exists():
                return seg_class_dir, item / "labelmap.txt"
    direct = line_folder / "SegmentationClass"
    if direct.exists():
        return direct, line_folder / "labelmap.txt"
    return None, None


def parse_labelmap(labelmap_path):
    """Parse CVAT labelmap.txt → dict {canonical_name: (R,G,B)}."""
    colors = {}
    with open(labelmap_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(":")
            if len(parts) < 2:
                continue
            label_name = parts[0].strip()
            color_str = parts[1].strip()
            if label_name.lower() == "background":
                continue
            try:
                r, g, b = [int(x) for x in color_str.split(",")]
                canonical = normalize_class_name(label_name)
                colors[canonical] = (r, g, b)
            except (ValueError, IndexError):
                continue
    return colors


def read_color_mask(mask_path, target_color, tolerance=30):
    """Read a CVAT color mask and extract binary mask for a color."""
    img = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    target = np.array(target_color, dtype=np.int32)
    diff = np.abs(img_rgb.astype(np.int32) - target)
    mask = np.all(diff <= tolerance, axis=2).astype(np.uint8)
    return mask


# ─── XML Rasterization (for Leader_Line etc.) ────────────────────────────────

def parse_points_string(points_str):
    """Parse CVAT points string 'x1,y1;x2,y2;...' → numpy array."""
    pts = []
    for pair in points_str.strip().split(";"):
        x, y = pair.split(",")
        pts.append([float(x), float(y)])
    return np.array(pts, dtype=np.float32)


def decode_rle_mask(rle_str, width, height, left, top, img_width, img_height):
    """Decode CVAT RLE mask into a full-image binary mask."""
    counts = [int(x) for x in rle_str.strip().split(",")]
    # RLE alternates between 0-runs and 1-runs, starting with 0
    flat = []
    val = 0
    for c in counts:
        flat.extend([val] * c)
        val = 1 - val

    # Reshape to local region
    local_mask = np.array(flat[:width * height], dtype=np.uint8).reshape(height, width)

    # Place into full image
    full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    y_end = min(top + height, img_height)
    x_end = min(left + width, img_width)
    h_clip = y_end - top
    w_clip = x_end - left
    full_mask[top:y_end, left:x_end] = local_mask[:h_clip, :w_clip]
    return full_mask


def rasterize_xml_annotations(xml_path, target_label_canonical, raw_images_dict, line_thickness=3):
    """
    Parse CVAT annotations.xml and rasterize polygon/polyline/mask
    annotations into binary masks per frame.

    Returns: dict {frame_name: binary_mask}
    """
    logger.info(f"  Rasterizing XML: {xml_path}")
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    # Discover all label names in this XML that map to our target class
    target_labels = set()
    for label_elem in root.findall(".//label"):
        name_elem = label_elem.find("name")
        if name_elem is not None:
            canonical = normalize_class_name(name_elem.text)
            if canonical == target_label_canonical:
                target_labels.add(name_elem.text.strip())

    if not target_labels:
        # Fallback: try all aliases
        for alias, canonical in NAME_ALIASES.items():
            if canonical == target_label_canonical:
                target_labels.add(alias)

    logger.info(f"  Target labels in XML: {target_labels}")

    masks = {}
    for image_elem in root.findall("image"):
        frame_name = Path(image_elem.get("name")).stem
        img_width = int(image_elem.get("width"))
        img_height = int(image_elem.get("height"))

        frame_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        has_annotations = False

        for child in image_elem:
            label = child.get("label", "").strip()
            if label not in target_labels:
                continue

            tag = child.tag

            if tag == "polygon":
                pts_str = child.get("points", "")
                if pts_str:
                    pts = parse_points_string(pts_str)
                    pts_int = pts.astype(np.int32)
                    cv2.fillPoly(frame_mask, [pts_int], 1)
                    has_annotations = True

            elif tag == "polyline":
                pts_str = child.get("points", "")
                if pts_str:
                    pts = parse_points_string(pts_str)
                    pts_int = pts.astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame_mask, [pts_int], isClosed=False,
                                  color=1, thickness=line_thickness)
                    has_annotations = True

            elif tag == "mask":
                rle = child.get("rle", "")
                left = int(child.get("left", 0))
                top = int(child.get("top", 0))
                w = int(child.get("width", 0))
                h = int(child.get("height", 0))
                if rle and w > 0 and h > 0:
                    rle_mask = decode_rle_mask(rle, w, h, left, top, img_width, img_height)
                    frame_mask = np.maximum(frame_mask, rle_mask)
                    has_annotations = True

        if has_annotations:
            masks[frame_name] = frame_mask

    logger.info(f"  Rasterized {len(masks)} frames from XML")
    return masks


# ─── Merge All Sources ────────────────────────────────────────────────────────

def build_merged_masks(annotation_dir, line_types, selected_lines=None):
    """
    Build multi-class label maps by merging per-line-type masks.
    Uses SegmentationClass masks where available, falls back to XML rasterization.
    """
    annotation_dir = Path(annotation_dir)
    raw_dir = find_raw_images_dir(annotation_dir)
    if raw_dir is None:
        logger.error("Could not find raw engineering drawings directory!")
        sys.exit(1)
    logger.info(f"Raw images directory: {raw_dir}")

    raw_images = {}
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif"):
        for img_path in raw_dir.glob(ext):
            raw_images[img_path.stem] = img_path
    logger.info(f"Found {len(raw_images)} raw images")

    if selected_lines:
        active_lines = {k: v for k, v in line_types.items() if k in selected_lines}
        logger.info(f"Selected {len(active_lines)} line types: {list(active_lines.keys())}")
    else:
        active_lines = line_types

    merged = {}
    class_frame_counts = {}  # Track how many frames each class has

    for class_name, folder_path in active_lines.items():
        class_id = MASTER_CLASSES[class_name]
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing: {class_name} (class {class_id})")

        # ── Source 1: Try SegmentationClass masks ──
        seg_dir, labelmap_path = find_segmentation_mask_dir(folder_path)
        used_seg_masks = False

        if seg_dir is not None and labelmap_path is not None and labelmap_path.exists():
            colors = parse_labelmap(labelmap_path)
            target_color = colors.get(class_name)

            if target_color is None:
                # Try any non-background color
                for cn, col in colors.items():
                    if cn != "background" and cn != "Background":
                        target_color = col
                        logger.info(f"  Using color from label '{cn}': {col}")
                        break

            if target_color is not None:
                logger.info(f"  [SEG MASK] Target color: RGB{target_color}")
                mask_files = sorted(seg_dir.glob("*.png"))
                processed = 0

                for mask_path in mask_files:
                    frame_name = mask_path.stem
                    if frame_name not in raw_images:
                        continue

                    binary_mask = read_color_mask(mask_path, target_color)
                    if binary_mask is None or binary_mask.sum() == 0:
                        continue

                    if frame_name not in merged:
                        raw_img = cv2.imread(str(raw_images[frame_name]))
                        h, w = raw_img.shape[:2]
                        merged[frame_name] = {
                            "image_path": raw_images[frame_name],
                            "mask": np.zeros((h, w), dtype=np.uint8),
                            "classes_present": set(),
                            "image_size": (h, w),
                        }

                    info = merged[frame_name]
                    h, w = info["image_size"]
                    if binary_mask.shape != (h, w):
                        binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                    info["mask"][binary_mask > 0] = class_id
                    info["classes_present"].add(class_name)
                    processed += 1

                if processed > 0:
                    used_seg_masks = True
                    class_frame_counts[class_name] = processed
                    logger.info(f"  [SEG MASK] Processed {processed} frames")

        # ── Source 2: Fall back to XML rasterization ──
        if not used_seg_masks:
            xml_path = folder_path / "annotations.xml"
            if xml_path.exists():
                logger.info(f"  [XML] No seg masks found, rasterizing from XML...")
                xml_masks = rasterize_xml_annotations(
                    xml_path, class_name, raw_images, line_thickness=3
                )

                processed = 0
                for frame_name, binary_mask in xml_masks.items():
                    if frame_name not in raw_images:
                        continue
                    if binary_mask.sum() == 0:
                        continue

                    if frame_name not in merged:
                        raw_img = cv2.imread(str(raw_images[frame_name]))
                        h, w = raw_img.shape[:2]
                        merged[frame_name] = {
                            "image_path": raw_images[frame_name],
                            "mask": np.zeros((h, w), dtype=np.uint8),
                            "classes_present": set(),
                            "image_size": (h, w),
                        }

                    info = merged[frame_name]
                    h, w = info["image_size"]
                    if binary_mask.shape != (h, w):
                        binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                    info["mask"][binary_mask > 0] = class_id
                    info["classes_present"].add(class_name)
                    processed += 1

                class_frame_counts[class_name] = processed
                logger.info(f"  [XML] Processed {processed} frames")
            else:
                logger.warning(f"  No seg masks AND no annotations.xml found!")
                class_frame_counts[class_name] = 0

        # ── Also check XML for additional frames not in seg masks ──
        if used_seg_masks:
            xml_path = folder_path / "annotations.xml"
            if xml_path.exists():
                xml_masks = rasterize_xml_annotations(
                    xml_path, class_name, raw_images, line_thickness=3
                )
                extra = 0
                for frame_name, binary_mask in xml_masks.items():
                    if frame_name not in raw_images:
                        continue
                    # Only use XML for frames NOT already covered by seg masks
                    if frame_name in merged and class_name in merged[frame_name]["classes_present"]:
                        continue
                    if binary_mask.sum() == 0:
                        continue

                    if frame_name not in merged:
                        raw_img = cv2.imread(str(raw_images[frame_name]))
                        h, w = raw_img.shape[:2]
                        merged[frame_name] = {
                            "image_path": raw_images[frame_name],
                            "mask": np.zeros((h, w), dtype=np.uint8),
                            "classes_present": set(),
                            "image_size": (h, w),
                        }

                    info = merged[frame_name]
                    h, w = info["image_size"]
                    if binary_mask.shape != (h, w):
                        binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                    info["mask"][binary_mask > 0] = class_id
                    info["classes_present"].add(class_name)
                    extra += 1

                if extra > 0:
                    class_frame_counts[class_name] += extra
                    logger.info(f"  [XML EXTRA] Added {extra} frames from XML not in seg masks")

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Total frames with annotations: {len(merged)}")
    logger.info(f"\nPer-class frame counts:")
    for cls, cnt in sorted(class_frame_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {cls:<25s} {cnt:>3d} frames")

    for fname, info in sorted(merged.items()):
        classes = ", ".join(sorted(info["classes_present"]))
        logger.info(f"  {fname}: classes=[{classes}]")

    return merged, class_frame_counts


# ─── Patch Extraction ────────────────────────────────────────────────────────

def extract_patches(merged_data, output_dir, patch_size=512, overlap=0.5,
                    min_fg_ratio=0.01, val_ratio=0.2, seed=42):
    """Extract foreground-aware patches with train/val split at image level."""
    output_dir = Path(output_dir)
    random.seed(seed)

    frame_names = sorted(merged_data.keys())
    random.shuffle(frame_names)
    n_val = max(1, int(len(frame_names) * val_ratio))
    val_frames = set(frame_names[:n_val])
    train_frames = set(frame_names[n_val:])

    logger.info(f"Split: {len(train_frames)} train, {len(val_frames)} val frames")

    stride = int(patch_size * (1 - overlap))
    stats = {"train": 0, "val": 0, "skipped": 0}
    metadata_rows = []
    # Track which classes appear in which patches for augmentation
    class_patch_counts = defaultdict(int)

    for split_name, frame_set in [("train", train_frames), ("val", val_frames)]:
        split_img_dir = output_dir / split_name / "images"
        split_mask_dir = output_dir / split_name / "masks"
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_mask_dir.mkdir(parents=True, exist_ok=True)

        for frame_name in tqdm(sorted(frame_set), desc=f"Extracting {split_name}"):
            info = merged_data[frame_name]
            image = cv2.imread(str(info["image_path"]))
            mask = info["mask"]

            if image is None:
                continue

            h, w = image.shape[:2]
            patch_idx = 0

            # Generate all y,x positions (regular grid + edge coverage)
            positions = set()
            for y in range(0, max(1, h - patch_size + 1), stride):
                for x in range(0, max(1, w - patch_size + 1), stride):
                    positions.add((y, x))
            # Edge positions
            if h >= patch_size:
                for x in range(0, max(1, w - patch_size + 1), stride):
                    positions.add((h - patch_size, x))
            if w >= patch_size:
                for y in range(0, max(1, h - patch_size + 1), stride):
                    positions.add((y, w - patch_size))
            if h >= patch_size and w >= patch_size:
                positions.add((h - patch_size, w - patch_size))

            for (y, x) in sorted(positions):
                y2 = min(y + patch_size, h)
                x2 = min(x + patch_size, w)
                if (y2 - y) < patch_size or (x2 - x) < patch_size:
                    continue

                img_patch = image[y:y2, x:x2]
                mask_patch = mask[y:y2, x:x2]

                fg_ratio = (mask_patch > 0).sum() / (patch_size * patch_size)
                if fg_ratio < min_fg_ratio:
                    stats["skipped"] += 1
                    continue

                classes_in_patch = sorted(set(int(c) for c in mask_patch[mask_patch > 0].tolist()))
                patch_name = f"{frame_name}_p{patch_idx:04d}"

                cv2.imwrite(str(split_img_dir / f"{patch_name}.png"), img_patch)
                cv2.imwrite(str(split_mask_dir / f"{patch_name}.png"), mask_patch)

                metadata_rows.append({
                    "filename": f"{patch_name}.png",
                    "split": split_name,
                    "source_frame": frame_name,
                    "x": x, "y": y,
                    "fg_ratio": round(float(fg_ratio), 4),
                    "classes": json.dumps(classes_in_patch),
                    "n_classes": len(classes_in_patch),
                })

                if split_name == "train":
                    for c in classes_in_patch:
                        class_patch_counts[c] += 1

                stats[split_name] += 1
                patch_idx += 1

    # Save metadata
    meta_path = output_dir / "metadata.csv"
    if metadata_rows:
        with open(meta_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metadata_rows[0].keys())
            writer.writeheader()
            writer.writerows(metadata_rows)

    return stats, class_patch_counts


# ─── Augmentation Balancing ───────────────────────────────────────────────────

def get_augmentation_transform():
    """Strong augmentation for balancing."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, value=0,
                 mask_value=0, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.GaussNoise(p=0.3),
    ])


def compute_augmentation_multipliers(class_patch_counts, max_count=None):
    """
    Compute per-class augmentation multipliers.
    Target: bring all classes to at least 70% of the max class count.
    """
    if not class_patch_counts:
        return {}

    if max_count is None:
        max_count = max(class_patch_counts.values())

    target_count = int(max_count * 0.7)
    multipliers = {}

    for cls_id, count in class_patch_counts.items():
        if cls_id == 0:  # skip background
            continue
        if count >= target_count:
            multipliers[cls_id] = 0  # No extra augmentation needed
        elif count > 0:
            needed = target_count - count
            mult = min(int(np.ceil(needed / count)), 8)  # Cap at 8×
            multipliers[cls_id] = mult
        else:
            multipliers[cls_id] = 0

    return multipliers


def augment_underrepresented_patches(output_dir, class_patch_counts, id_to_name):
    """
    Create augmented copies of patches from underrepresented classes.
    Only augments TRAIN patches.
    """
    output_dir = Path(output_dir)
    train_img_dir = output_dir / "train" / "images"
    train_mask_dir = output_dir / "train" / "masks"

    multipliers = compute_augmentation_multipliers(class_patch_counts)

    logger.info("\nAugmentation multipliers:")
    for cls_id, mult in sorted(multipliers.items()):
        name = id_to_name.get(cls_id, f"class_{cls_id}")
        count = class_patch_counts.get(cls_id, 0)
        logger.info(f"  {name:<25s} {count:>4d} patches → +{mult}× augmented")

    # Find which patches contain each underrepresented class
    classes_needing_aug = {c for c, m in multipliers.items() if m > 0}
    if not classes_needing_aug:
        logger.info("  No augmentation needed — all classes are balanced.")
        return 0

    transform = get_augmentation_transform()
    augmented_count = 0

    # Read metadata to find patches per class
    meta_path = output_dir / "metadata.csv"
    if not meta_path.exists():
        logger.warning("  No metadata.csv found, skipping augmentation")
        return 0

    import pandas as pd
    df = pd.read_csv(meta_path)
    train_df = df[df["split"] == "train"]

    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Augmenting"):
        patch_classes = json.loads(row["classes"])
        # Check if this patch has any class that needs augmentation
        needs_aug = [c for c in patch_classes if c in classes_needing_aug]
        if not needs_aug:
            continue

        # Use the maximum multiplier among the classes in this patch
        max_mult = max(multipliers.get(c, 0) for c in needs_aug)
        if max_mult <= 0:
            continue

        fname = row["filename"]
        img_path = train_img_dir / fname
        mask_path = train_mask_dir / fname

        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue

        base_name = fname.replace(".png", "")

        for aug_idx in range(max_mult):
            transformed = transform(image=img, mask=mask)
            aug_img = transformed["image"]
            aug_mask = transformed["mask"]

            aug_name = f"{base_name}_aug{aug_idx:02d}.png"
            cv2.imwrite(str(train_img_dir / aug_name), aug_img)
            cv2.imwrite(str(train_mask_dir / aug_name), aug_mask)
            augmented_count += 1

    logger.info(f"  Generated {augmented_count} augmented patches")
    return augmented_count


# ─── Previews ─────────────────────────────────────────────────────────────────

PREVIEW_COLORS = [
    (0, 0, 0),        # 0: Background
    (255, 0, 0),      # 1: Center_line
    (250, 50, 83),    # 2: Dimension_lines
    (139, 233, 87),   # 3: Extension_line
    (0, 255, 255),    # 4: Feature_Visible
    (255, 165, 0),    # 5: Leader_line
    (128, 0, 255),    # 6: Phantom_lines
    (255, 255, 0),    # 7: break_line
    (0, 128, 255),    # 8: cutting_plane
    (255, 0, 255),    # 9: hidden_lines
    (0, 200, 100),    # 10: Section_hatching
]


def save_preview(merged_data, output_dir, max_previews=8):
    """Save preview images showing merged masks overlaid on raw images."""
    output_dir = Path(output_dir) / "previews"
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for frame_name, info in sorted(merged_data.items()):
        if count >= max_previews:
            break

        image = cv2.imread(str(info["image_path"]))
        if image is None:
            continue

        mask = info["mask"]
        overlay = image.copy()

        for class_id in range(1, max(MASTER_CLASSES.values()) + 1):
            class_pixels = mask == class_id
            if class_pixels.any():
                color = PREVIEW_COLORS[class_id] if class_id < len(PREVIEW_COLORS) else (128, 128, 128)
                overlay[class_pixels] = color

        blended = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)

        y_offset = 30
        for class_id in sorted(set(int(c) for c in mask[mask > 0].tolist())):
            name = "Unknown"
            for n, cid in MASTER_CLASSES.items():
                if cid == class_id:
                    name = n
                    break
            color = PREVIEW_COLORS[class_id] if class_id < len(PREVIEW_COLORS) else (128, 128, 128)
            cv2.putText(blended, f"{class_id}: {name}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25

        cv2.imwrite(str(output_dir / f"{frame_name}_preview.jpg"), blended)
        count += 1

    logger.info(f"Saved {count} preview images to {output_dir}")


def save_class_mapping(output_dir, active_classes):
    """Save class mapping for this dataset."""
    output_dir = Path(output_dir)
    mapping = {"Background": 0}
    for cls_name in sorted(active_classes):
        mapping[cls_name] = MASTER_CLASSES[cls_name]
    out_path = output_dir / "classes.json"
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2)
    logger.info(f"Class mapping saved to {out_path}")
    return mapping


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare CVAT annotation dataset (v2)")
    parser.add_argument("--annotation_dir", type=str,
                        default=r"C:\Users\Admin\line_detection\Approach 2\Annotation",
                        help="Root annotation directory")
    parser.add_argument("--output_dir", type=str,
                        default=r"C:\Users\Admin\line_detection\Approach 2\annotation_patches",
                        help="Output directory for patches")
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--min_fg_ratio", type=float, default=0.01)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--lines", nargs="+", default=None,
                        help="Specific line types to include. "
                             "Example: --lines Dimension_lines Extension_line Feature_Visible")
    parser.add_argument("--no_augment", action="store_true",
                        help="Disable augmentation balancing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("CVAT Annotation Dataset Preparation (v2)")
    logger.info("=" * 60)
    logger.info(f"Annotation dir: {args.annotation_dir}")
    logger.info(f"Output dir:     {args.output_dir}")
    logger.info(f"Patch size:     {args.patch_size}")
    logger.info(f"Overlap:        {args.overlap}")
    logger.info(f"Min FG ratio:   {args.min_fg_ratio}")

    # Build ID→name mapping
    id_to_name = {v: k for k, v in MASTER_CLASSES.items()}

    # Step 1: Discover
    logger.info("\n--- Step 1: Discovering line types ---")
    line_types = discover_line_types(args.annotation_dir)
    if not line_types:
        logger.error("No line type folders found!")
        sys.exit(1)

    selected_lines = None
    if args.lines:
        selected_lines = [normalize_class_name(l) for l in args.lines]
        logger.info(f"Selected lines: {selected_lines}")

    # Step 2: Build merged masks
    logger.info("\n--- Step 2: Building merged masks ---")
    merged_data, class_frame_counts = build_merged_masks(
        args.annotation_dir, line_types, selected_lines
    )
    if not merged_data:
        logger.error("No annotated frames found!")
        sys.exit(1)

    # Step 3: Save previews
    logger.info("\n--- Step 3: Saving previews ---")
    save_preview(merged_data, args.output_dir)

    # Step 4: Extract patches
    logger.info("\n--- Step 4: Extracting patches ---")
    stats, class_patch_counts = extract_patches(
        merged_data, args.output_dir,
        patch_size=args.patch_size,
        overlap=args.overlap,
        min_fg_ratio=args.min_fg_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    logger.info(f"\nTrain patches per class (before augmentation):")
    for cls_id, count in sorted(class_patch_counts.items()):
        name = id_to_name.get(cls_id, f"class_{cls_id}")
        logger.info(f"  {name:<25s} {count:>4d} patches")

    # Step 5: Augmentation balancing
    aug_count = 0
    if not args.no_augment and class_patch_counts:
        logger.info("\n--- Step 5: Augmentation balancing ---")
        aug_count = augment_underrepresented_patches(
            args.output_dir, class_patch_counts, id_to_name
        )

    # Step 6: Save class mapping
    logger.info("\n--- Step 6: Saving class mapping ---")
    all_classes = set()
    for info in merged_data.values():
        all_classes.update(info["classes_present"])
    class_mapping = save_class_mapping(args.output_dir, all_classes)

    master_path = Path(args.output_dir) / "master_classes.json"
    with open(master_path, "w") as f:
        json.dump(MASTER_CLASSES, f, indent=2)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DATASET PREPARATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Train patches:  {stats['train']} original + {aug_count} augmented = {stats['train'] + aug_count}")
    logger.info(f"  Val patches:    {stats['val']}")
    logger.info(f"  Skipped (low FG): {stats['skipped']}")
    logger.info(f"  Active classes: {class_mapping}")
    logger.info(f"  Output:         {args.output_dir}")
    logger.info(f"\nNext step:")
    logger.info(f"  python finetune_annotation.py --num_classes {max(MASTER_CLASSES.values()) + 1}")


if __name__ == "__main__":
    main()
