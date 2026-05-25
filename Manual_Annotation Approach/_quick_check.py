"""Quick check on cleaned dataset: sizes, mask encoding, sample stats."""
import os, json, numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter

root = Path(r"c:\Users\Admin\line_detection\Approach 2\pipeline_output")
img_dir = root / "cleaned" / "images"
mask_dir = root / "cleaned" / "masks"

imgs = sorted(img_dir.glob("*.png"))
print(f"Total images: {len(imgs)}")

# Check first 20 samples for size/format
sizes = Counter()
for p in imgs[:100]:
    im = Image.open(p)
    sizes[im.size] += 1

print(f"Image size distribution (first 100): {dict(sizes)}")

# Check masks for valid class IDs
with open(root / "classes_cleaned.json") as f:
    cmap = json.load(f)
print(f"Class map: {cmap}")

valid_ids = set([0] + list(cmap.values()))
print(f"Valid mask IDs: {valid_ids}")

# Check 20 random masks
import random
random.seed(42)
sample = random.sample(list(imgs), min(20, len(imgs)))
for p in sample[:5]:
    mp = mask_dir / p.name
    if mp.exists():
        m = np.array(Image.open(mp))
        uniq = np.unique(m)
        fg_ratio = np.count_nonzero(m) / m.size * 100
        print(f"  {p.name}: shape={m.shape}, unique={uniq}, fg={fg_ratio:.2f}%")

# Check split files
for split_name in ["train", "val", "test"]:
    sp = root / "splits" / f"{split_name}.txt"
    with open(sp) as f:
        stems = [l.strip() for l in f if l.strip()]
    print(f"{split_name}: {len(stems)} stems, sample: {stems[0]}")
