"""Quick exploration script to understand dataset structure."""
import os
from pathlib import Path
from collections import defaultdict

# 1. Count files in the Essential dataset (single-line images)
essential_root = Path(r"c:\Users\Admin\line_detection\output\dataset_cleaned\Essential")
print("=" * 60)
print("ESSENTIAL DATASET (Single-line support data)")
print("=" * 60)
for class_dir in sorted(essential_root.iterdir()):
    if class_dir.is_dir():
        img_dir = class_dir / "images"
        mask_dir = class_dir / "masks"
        img_count = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        mask_count = len(list(mask_dir.glob("*"))) if mask_dir.exists() else 0
        print(f"  {class_dir.name:40s} images={img_count:5d}  masks={mask_count:5d}")
        # Show a sample filename
        if img_count > 0:
            sample = list(img_dir.iterdir())[0]
            print(f"    sample image: {sample.name}")
        if mask_count > 0:
            sample = list(mask_dir.iterdir())[0]
            print(f"    sample mask: {sample.name}")

# 2. Engineering drawings - check if there are masks
eng_root = Path(r"c:\Users\Admin\line_detection\Approach 2\Engineering_Drawings")
print("\n" + "=" * 60)
print("ENGINEERING DRAWINGS (Real CAD data)")
print("=" * 60)
all_files = list(eng_root.iterdir())
extensions = defaultdict(int)
for f in all_files:
    if f.is_file():
        extensions[f.suffix.lower()] += 1
print(f"  Total files: {len(all_files)}")
print(f"  Extensions: {dict(extensions)}")

# Check for potential mask directories nearby
approach2_root = Path(r"c:\Users\Admin\line_detection\Approach 2")
print(f"\n  Approach 2 subdirectories:")
for item in sorted(approach2_root.iterdir()):
    if item.is_dir():
        sub_count = len(list(item.iterdir()))
        print(f"    {item.name}: {sub_count} items")

# 3. Check Non_essential dataset
non_ess_root = Path(r"c:\Users\Admin\line_detection\Dataset\Non_essential")
print("\n" + "=" * 60)
print("NON-ESSENTIAL DATASET")
print("=" * 60)
for class_dir in sorted(non_ess_root.iterdir()):
    if class_dir.is_dir():
        count = len(list(class_dir.glob("**/*")))
        print(f"  {class_dir.name:40s} total_files={count}")

# 4. Check existing training data/splits
print("\n" + "=" * 60)
print("EXISTING SPLITS")
print("=" * 60)
splits_dir = Path(r"c:\Users\Admin\line_detection\output\splits")
for f in splits_dir.iterdir():
    with open(f, 'r') as fh:
        lines = fh.readlines()
    print(f"  {f.name}: {len(lines)} entries")
    if lines:
        print(f"    first entry: {lines[0].strip()}")
        print(f"    last entry: {lines[-1].strip()}")

# 5. Check existing training scripts for class mapping info
print("\n" + "=" * 60)
print("SEARCHING FOR CLASS MAPPINGS IN EXISTING CODE")
print("=" * 60)

for pyfile in Path(r"c:\Users\Admin\line_detection").glob("*.py"):
    with open(pyfile, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    if 'class_map' in content.lower() or 'CLASS_NAMES' in content or 'class_to_id' in content.lower():
        print(f"\n  Found class mapping references in: {pyfile.name}")
        for i, line in enumerate(content.split('\n')):
            if any(kw in line.lower() for kw in ['class_map', 'class_names', 'class_to_id', 'id_to_class']):
                print(f"    L{i+1}: {line.strip()[:120]}")
