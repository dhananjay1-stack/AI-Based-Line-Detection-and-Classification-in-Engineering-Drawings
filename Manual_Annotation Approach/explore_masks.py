"""Explore mask format and encoding in detail."""
import os
import numpy as np
from pathlib import Path
from PIL import Image

essential_root = Path(r"c:\Users\Admin\line_detection\output\dataset_cleaned\Essential")

# Check mask encoding for each class
print("=" * 80)
print("MASK FORMAT ANALYSIS (Essential single-line dataset)")
print("=" * 80)

for class_dir in sorted(essential_root.iterdir()):
    if not class_dir.is_dir():
        continue
    mask_dir = class_dir / "masks"
    img_dir = class_dir / "images"
    if not mask_dir.exists():
        continue
    
    masks = sorted(list(mask_dir.glob("*")))[:3]  # Check first 3
    images = sorted(list(img_dir.glob("*")))[:1]
    
    print(f"\n--- {class_dir.name} ---")
    for m in masks:
        try:
            mask = Image.open(m)
            arr = np.array(mask)
            print(f"  {m.name}: mode={mask.mode}, shape={arr.shape}, dtype={arr.dtype}, "
                  f"unique_values={np.unique(arr).tolist()[:20]}, "
                  f"min={arr.min()}, max={arr.max()}")
        except Exception as e:
            print(f"  {m.name}: ERROR - {e}")
    
    # Also check image format
    if images:
        img = Image.open(images[0])
        arr = np.array(img)
        print(f"  IMAGE sample: mode={img.mode}, shape={arr.shape}, dtype={arr.dtype}")

# Check if masks are binary (single class per mask) or multi-class
print("\n" + "=" * 80)
print("MASK VALUE DISTRIBUTION (deeper check)")
print("=" * 80)

for class_dir in sorted(essential_root.iterdir()):
    if not class_dir.is_dir():
        continue
    mask_dir = class_dir / "masks"
    if not mask_dir.exists():
        continue
    
    masks = list(mask_dir.glob("*"))
    # Check 10 random masks
    import random
    random.seed(42)
    sample_masks = random.sample(masks, min(10, len(masks)))
    
    all_unique = set()
    channels = set()
    modes = set()
    
    for m in sample_masks:
        try:
            mask = Image.open(m)
            arr = np.array(mask)
            modes.add(mask.mode)
            if len(arr.shape) == 2:
                channels.add(1)
            else:
                channels.add(arr.shape[2])
            all_unique.update(np.unique(arr).tolist())
        except:
            pass
    
    print(f"  {class_dir.name:40s} modes={modes}, channels={channels}, "
          f"unique_values={sorted(all_unique)[:30]}")

# Check the Non_essential dataset structure too
print("\n" + "=" * 80)
print("NON-ESSENTIAL DATASET STRUCTURE")
print("=" * 80)
non_ess = Path(r"c:\Users\Admin\line_detection\Dataset\Non_essential")
for class_dir in sorted(non_ess.iterdir()):
    if not class_dir.is_dir():
        continue
    subdirs = [d.name for d in class_dir.iterdir() if d.is_dir()]
    files = [f.name for f in class_dir.iterdir() if f.is_file()][:3]
    print(f"  {class_dir.name}: subdirs={subdirs}, sample_files={files}")

# Check the existing training code for how masks are loaded
print("\n" + "=" * 80)
print("CHECKING EXISTING DATA LOADING")
print("=" * 80)
train_py = Path(r"c:\Users\Admin\line_detection\train.py")
if train_py.exists():
    with open(train_py, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    # Find mask loading logic
    for i, line in enumerate(content.split('\n')):
        if any(kw in line.lower() for kw in ['mask', 'label', 'target', 'class_id', 'imread']):
            print(f"  train.py L{i+1}: {line.rstrip()[:120]}")
