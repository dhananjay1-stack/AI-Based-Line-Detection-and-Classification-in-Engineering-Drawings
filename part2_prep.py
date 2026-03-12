# part2_prep.py
#
# Usage:
#   pip install numpy opencv-python Pillow tqdm pandas scikit-image shapely
#
# Examples:
#   python part2_prep.py prepare_classes --dataset_root ./dataset
#   python part2_prep.py build_multiclass_masks --dataset_root ./dataset --out_dir ./output
#   python part2_prep.py create_tiles --tile_size 1024 --overlap 128 --oversample_factor 3
#   python part2_prep.py show_preview --preview_n 5

import os
import sys
import argparse
import json
import glob
import shutil
import random
import logging
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from PIL import Image

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants for file extensions
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def get_args():
    parser = argparse.ArgumentParser(description="Part-2 Dataset Preparation Tool")
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Operation mode')

    # Shared arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--dataset_root', type=str, default='C:/Users/Admin/line_detection/output/dataset_cleaned', help='Root of the dataset (contains Essential/)')
    parent_parser.add_argument('--out_dir', type=str, default='./output', help='Output directory for artifacts')
    parent_parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parent_parser.add_argument('--dry_run', action='store_true', help='Simulate operations without writing files')

    # Mode: prepare_classes
    parser_prep = subparsers.add_parser('prepare_classes', parents=[parent_parser], help='Scan classes and create splits')
    parser_prep.add_argument('--train_split', type=float, default=0.8, help='Fraction of data for training')
    parser_prep.add_argument('--val_split', type=float, default=0.2, help='Fraction of data for validation')

    # Mode: build_multiclass_masks
    parser_build = subparsers.add_parser('build_multiclass_masks', parents=[parent_parser], help='Merge binary masks into multiclass PNGs')

    # Mode: create_tiles
    parser_tile = subparsers.add_parser('create_tiles', parents=[parent_parser], help='Tile images and masks')
    parser_tile.add_argument('--tile_size', type=int, default=640, help='Size of tiles (WxH)')
    parser_tile.add_argument('--overlap', type=int, default=128, help='Overlap between tiles')
    parser_tile.add_argument('--min_mask_fraction_to_keep', type=float, default=0.001, help='Min mask content to keep a tile')
    parser_tile.add_argument('--oversample_factor', type=int, default=1, help='Copy tiles N times if they contain small objects')
    parser_tile.add_argument('--small_objects', type=str, default="arrowheads,tick_marks,text", help='Comma-separated class names to consider "small"')

    # Mode: show_preview
    parser_preview = subparsers.add_parser('show_preview', parents=[parent_parser], help='Generate side-by-side previews')
    parser_preview.add_argument('--preview_n', type=int, default=10, help='Number of preview images to generate')

    return parser.parse_args()

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def normalize_class_name(name):
    """Normalize directory names to class names (spaces to underscores)."""
    return name.strip().replace(" ", "_")

def scan_dataset(root_dir):
    """
    Scans the Essential/ folder.
    Returns:
        classes: List of normalized class names.
        image_map: Dict {filename: { 'path': full_path, 'class': class_name } }
                   (Note: If images are duplicated across class folders, this picks one source).
        mask_map: Dict {filename: { class_name: mask_path } }
    """
    essential_root = os.path.join(root_dir, 'Essential')
    if not os.path.exists(essential_root):
        logger.error(f"Directory not found: {essential_root}")
        sys.exit(1)

    found_classes = set()
    image_map = {} # filename -> info
    mask_map = defaultdict(dict) # filename -> {class: path}

    # Walk specific structure: Essential/<class>/images and Essential/<class>/masks
    subdirs = [d for d in os.listdir(essential_root) if os.path.isdir(os.path.join(essential_root, d))]
    
    for class_raw in subdirs:
        class_name = normalize_class_name(class_raw)
        found_classes.add(class_name)
        
        class_dir = os.path.join(essential_root, class_raw)
        img_dir = os.path.join(class_dir, 'images')
        msk_dir = os.path.join(class_dir, 'masks')

        # Scan Images
        if os.path.exists(img_dir):
            for f in os.listdir(img_dir):
                if os.path.splitext(f)[1].lower() in IMG_EXTS:
                    # We store the first valid image path we find for this filename
                    if f not in image_map:
                        image_map[f] = {'path': os.path.join(img_dir, f), 'primary_class': class_name}

        # Scan Masks
        if os.path.exists(msk_dir):
            for f in os.listdir(msk_dir):
                # Mask usually has same basename as image, usually png
                # If extension differs, we need to match by stem
                # For simplicity, assuming mask filename (incl ext) matches or we match by stem
                # Let's map by stem to be robust, but store full filename key
                mask_path = os.path.join(msk_dir, f)
                # Heuristic: assume mask filename correlates to image filename
                # We will attach this mask to the image key (filename)
                # If ext differs, we try to match stems
                image_key = f 
                # Attempt to find matching image key if strict match fails
                if image_key not in image_map:
                    stem = os.path.splitext(f)[0]
                    # Search image map for matching stem
                    for k in image_map.keys():
                        if os.path.splitext(k)[0] == stem:
                            image_key = k
                            break
                
                mask_map[image_key][class_name] = mask_path

    return sorted(list(found_classes)), image_map, mask_map

# -----------------------------------------------------------------------------
# Mode Implementations
# -----------------------------------------------------------------------------

def run_prepare_classes(args):
    """
    Mode 1: Discover classes, check pairs, split train/val.
    """
    logger.info("Scanning dataset for classes and checking integrity...")
    classes, image_map, mask_map = scan_dataset(args.dataset_root)
    
    logger.info(f"Found {len(classes)} classes: {classes}")
    
    # 1. Write classes.json
    class_dict = {name: idx + 1 for idx, name in enumerate(classes)} # 0 is background
    os.makedirs(args.out_dir, exist_ok=True)
    
    if not args.dry_run:
        with open(os.path.join(args.out_dir, 'classes.json'), 'w') as f:
            json.dump(class_dict, f, indent=4)
        logger.info(f"Saved classes.json to {args.out_dir}")

    # 2. Validate Pairs
    problems = []
    valid_images = []
    
    for fname, info in image_map.items():
        img_masks = mask_map.get(fname, {})
        if not img_masks:
            problems.append({'filename': fname, 'path': info['path'], 'issue': 'Missing mask for all classes'})
        else:
            valid_images.append(info['path'])

    # Report problems
    if problems:
        logger.warning(f"Found {len(problems)} issues. Writing to problems.csv")
        if not args.dry_run:
            pd.DataFrame(problems).to_csv(os.path.join(args.out_dir, 'problems.csv'), index=False)
    else:
        logger.info("No missing masks found.")

    # 3. Splits
    random.seed(42)
    random.shuffle(valid_images)
    num_train = int(len(valid_images) * args.train_split)
    train_files = valid_images[:num_train]
    val_files = valid_images[num_train:]

    splits_dir = os.path.join(args.out_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    if not args.dry_run:
        with open(os.path.join(splits_dir, 'train.txt'), 'w') as f:
            f.write('\n'.join(train_files))
        with open(os.path.join(splits_dir, 'val.txt'), 'w') as f:
            f.write('\n'.join(val_files))
        logger.info(f"Created splits: Train={len(train_files)}, Val={len(val_files)}")

def run_build_multiclass_masks(args):
    """
    Mode 2: Merge per-class masks into a single indexed PNG.
    """
    logger.info("Building multiclass masks...")
    
    # Load classes
    classes_json_path = os.path.join(args.out_dir, 'classes.json')
    if not os.path.exists(classes_json_path):
        logger.error("classes.json not found. Run 'prepare_classes' first.")
        sys.exit(1)
        
    with open(classes_json_path, 'r') as f:
        class_dict = json.load(f)
    
    # Invert dict for priority logging if needed, or just iterate keys
    # Priority is implied by alphabetical order in classes.json (value 1..N) unless specified otherwise.
    # We will respect the integer ID. Higher ID overwrites Lower ID? 
    # Usually small objects (higher ID often if alphabetical?) should be on top.
    # Let's ensure strict order: We iterate classes in specific order.
    # The requirement says: "priority by classes.json order". 
    # If 'Arrowhead' is 1 and 'Wall' is 2. If we paint 1 then 2, Wall covers Arrowhead.
    # Usually we want small things on top. 
    # Use alphabetical logic: paint in reverse order? Or just paint in order?
    # Let's stick to standard painter's algo: Iterate 1..N. Last one painted wins.
    # To be safe, usually 'Background' is 0.
    
    classes_list = sorted(class_dict.items(), key=lambda x: x[1])
    
    _, image_map, mask_map = scan_dataset(args.dataset_root)
    
    merged_dir = os.path.join(args.out_dir, 'merged_masks')
    os.makedirs(merged_dir, exist_ok=True)
    
    overlap_log = []

    for fname, info in tqdm(image_map.items(), desc="Merging Masks"):
        img_path = info['path']
        masks_avail = mask_map.get(fname, {})
        
        if not masks_avail:
            continue
            
        # Read reference image for shape
        # using PIL to avoid loading pixel data fully if possible, but we need shape
        with Image.open(img_path) as im:
            w, h = im.size
            
        # Create canvas (0 = background)
        # Using uint8 (supports up to 255 classes)
        canvas = np.zeros((h, w), dtype=np.uint8)
        
        # We iterate through the classes in the order defined in classes.json
        # The prompt says "priority by classes.json order".
        # If we paint 1, then 2... 2 overwrites 1.
        for cls_name, cls_id in classes_list:
            if cls_name in masks_avail:
                mask_path = masks_avail[cls_name]
                # Read mask as grayscale
                m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if m is None:
                    logger.warning(f"Could not read mask: {mask_path}")
                    continue
                
                # Resize if necessary (sanity check)
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                
                # Binarize
                _, bin_mask = cv2.threshold(m, 127, 1, cv2.THRESH_BINARY)
                
                # Check overlap
                overlap_check = (canvas > 0) & (bin_mask > 0)
                if np.any(overlap_check):
                    overlap_log.append(f"{fname}: Class '{cls_name}' (ID {cls_id}) overlapped existing pixels.")
                
                # Paint
                canvas[bin_mask == 1] = cls_id

        # Save merged mask
        # We save with the original filename but as .png
        out_name = os.path.splitext(fname)[0] + ".png"
        out_path = os.path.join(merged_dir, out_name)
        
        if not args.dry_run:
            # Save using PIL to ensure no compression artifacts on indices
            Image.fromarray(canvas).save(out_path)

    if overlap_log and args.verbose:
        logger.info(f"Encountered {len(overlap_log)} overlap events.")
        # Optionally write to file
        
    logger.info(f"Merged masks saved to {merged_dir}")

def run_create_tiles(args):
    """
    Mode 3: Tile images and masks, handle oversampling.
    """
    logger.info("Creating tiles...")
    
    # Config
    TILE_SIZE = args.tile_size
    OVERLAP = args.overlap
    STRIDE = TILE_SIZE - OVERLAP
    
    # Paths
    merged_mask_dir = os.path.join(args.out_dir, 'merged_masks')
    tiles_out = os.path.join("tiles") # per requirements, root ./tiles
    tiles_img_dir = os.path.join(tiles_out, 'images')
    tiles_msk_dir = os.path.join(tiles_out, 'masks')
    
    if not args.dry_run:
        os.makedirs(tiles_img_dir, exist_ok=True)
        os.makedirs(tiles_msk_dir, exist_ok=True)
    
    # Load Classes to identify small objects
    classes_json_path = os.path.join(args.out_dir, 'classes.json')
    with open(classes_json_path, 'r') as f:
        class_dict = json.load(f)
        
    small_obj_names = [s.strip() for s in args.small_objects.split(',')]
    small_obj_ids = [class_dict[n] for n in small_obj_names if n in class_dict]
    logger.info(f"Oversampling logic active for IDs: {small_obj_ids} (Factor: {args.oversample_factor})")

    # Get list of merged masks to match with images
    # We rely on image_map to find original images, then look for merged mask
    _, image_map, _ = scan_dataset(args.dataset_root)
    
    metadata = []
    
    for fname, info in tqdm(image_map.items(), desc="Tiling"):
        img_path = info['path']
        # Find corresponding merged mask
        mask_name = os.path.splitext(fname)[0] + ".png"
        mask_path = os.path.join(merged_mask_dir, mask_name)
        
        if not os.path.exists(mask_path):
            # Might happen if mask generation skipped due to error
            continue
            
        # Load Image and Mask
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Working in RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # 2D array
        
        h, w, _ = img.shape
        
        # Sliding Window
        for y in range(0, h, STRIDE):
            for x in range(0, w, STRIDE):
                y_end = min(h, y + TILE_SIZE)
                x_end = min(w, x + TILE_SIZE)
                
                # Actual crop coordinates
                y_start = y
                x_start = x
                
                # Adjust if tile is smaller than TILE_SIZE (edges)
                # Option A: Pad. Option B: Shift back.
                # Here we shift back to ensure fixed tile size unless image is smaller than tile
                if (y_end - y_start) < TILE_SIZE and h >= TILE_SIZE:
                    y_start = h - TILE_SIZE
                    y_end = h
                if (x_end - x_start) < TILE_SIZE and w >= TILE_SIZE:
                    x_start = w - TILE_SIZE
                    x_end = w
                    
                # If image is smaller than tile size, we pad
                img_tile = img[y_start:y_end, x_start:x_end]
                mask_tile = mask[y_start:y_end, x_start:x_end]
                
                # Padding logic if needed
                cur_h, cur_w = mask_tile.shape
                if cur_h < TILE_SIZE or cur_w < TILE_SIZE:
                    pad_h = TILE_SIZE - cur_h
                    pad_w = TILE_SIZE - cur_w
                    img_tile = np.pad(img_tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
                    mask_tile = np.pad(mask_tile, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
                
                # Check Content
                # Count non-zero pixels
                mask_area = np.count_nonzero(mask_tile)
                total_area = TILE_SIZE * TILE_SIZE
                fraction = mask_area / total_area
                
                if fraction < args.min_mask_fraction_to_keep:
                    continue
                
                # Identify classes present
                unique_classes = np.unique(mask_tile)
                unique_classes = unique_classes[unique_classes != 0] # remove bg
                classes_present_str = str(list(unique_classes))
                
                # Check Oversampling
                count = 1
                is_small = False
                if any(uid in unique_classes for uid in small_obj_ids):
                    count = args.oversample_factor
                    is_small = True
                    
                # Save
                tile_base_name = f"{os.path.splitext(fname)[0]}_{y_start}_{x_start}"
                
                for c in range(count):
                    suffix = f"_copy{c}" if c > 0 else ""
                    tile_fname = f"{tile_base_name}{suffix}.png" # Save tiles as PNG
                    
                    if not args.dry_run:
                        # Save Image
                        Image.fromarray(img_tile).save(os.path.join(tiles_img_dir, tile_fname))
                        # Save Mask (keep raw values)
                        Image.fromarray(mask_tile).save(os.path.join(tiles_msk_dir, tile_fname))
                    
                    metadata.append({
                        'parent_image': fname,
                        'tile_filename': tile_fname,
                        'x': x_start, 'y': y_start,
                        'mask_fraction': fraction,
                        'classes': classes_present_str,
                        'is_small_object': is_small
                    })
                    
    if not args.dry_run and metadata:
        pd.DataFrame(metadata).to_csv(os.path.join(tiles_out, 'tiles_metadata.csv'), index=False)
        logger.info(f"Created {len(metadata)} tiles. Metadata saved.")

def run_show_preview(args):
    """
    Mode 4: Visualization.
    """
    logger.info("Generating previews...")
    
    merged_mask_dir = os.path.join(args.out_dir, 'merged_masks')
    preview_dir = os.path.join(args.out_dir, 'preview')
    os.makedirs(preview_dir, exist_ok=True)
    
    # Load classes for color map
    classes_json_path = os.path.join(args.out_dir, 'classes.json')
    if os.path.exists(classes_json_path):
        with open(classes_json_path) as f:
            class_dict = json.load(f)
            num_classes = len(class_dict) + 1
    else:
        num_classes = 20 # Fallback
        
    # Generate random colors
    colors = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0] # BG is black
    
    _, image_map, _ = scan_dataset(args.dataset_root)
    
    # Pick random samples
    all_files = list(image_map.keys())
    if not all_files:
        logger.error("No images found.")
        return

    samples = random.sample(all_files, min(len(all_files), args.preview_n))
    
    for fname in samples:
        img_path = image_map[fname]['path']
        mask_name = os.path.splitext(fname)[0] + ".png"
        mask_path = os.path.join(merged_mask_dir, mask_name)
        
        if not os.path.exists(mask_path):
            continue
            
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            continue
            
        # Create Color Mask
        color_mask = np.zeros_like(img)
        for cls_id in range(1, num_classes):
            color_mask[mask == cls_id] = colors[cls_id]
            
        # Blend
        overlay = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)
        
        # Concatenate side-by-side
        combined = np.hstack((img, overlay))
        
        out_path = os.path.join(preview_dir, f"preview_{fname}.jpg")
        if not args.dry_run:
            cv2.imwrite(out_path, combined)
            
    logger.info(f"Previews saved to {preview_dir}")

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main():
    args = get_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        if args.mode == 'prepare_classes':
            run_prepare_classes(args)
        elif args.mode == 'build_multiclass_masks':
            run_build_multiclass_masks(args)
        elif args.mode == 'create_tiles':
            run_create_tiles(args)
        elif args.mode == 'show_preview':
            run_show_preview(args)
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()