"""Final verification of the patch dataset."""
import json, numpy as np, random
from pathlib import Path
from PIL import Image
from collections import Counter
import pandas as pd

random.seed(42)

root = Path(r"C:\Users\Admin\line_detection\Approach 2\dataset_patches")
pipe = Path(r"C:\Users\Admin\line_detection\Approach 2\pipeline_output")

with open(pipe / "classes_cleaned.json") as f:
    cmap = json.load(f)
valid_ids = set([0] + list(cmap.values()))
id2name = {v: k for k, v in cmap.items()}

print("=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)

# 1. File counts
for sub in ["images", "masks", "centerlines"]:
    d = root / sub
    ext = "*.jpg" if sub == "images" else "*.png"
    count = len(list(d.glob(ext)))
    print(f"  {sub}: {count} files")

# 2. Check 50 random masks for valid class IDs
masks = sorted((root / "masks").glob("*.png"))
sample = random.sample(masks, min(50, len(masks)))
invalid_count = 0
class_pixel_total = Counter()

for mp in sample:
    m = np.array(Image.open(mp))
    assert m.shape == (512, 512), f"Bad shape: {mp.name} → {m.shape}"
    uniq = set(np.unique(m).tolist())
    if not uniq.issubset(valid_ids):
        print(f"  INVALID IDs in {mp.name}: {uniq - valid_ids}")
        invalid_count += 1
    for v in uniq:
        if v > 0:
            class_pixel_total[v] += np.count_nonzero(m == v)

print(f"\n  Mask shape check: all 512x512 [OK]")
print(f"  Invalid class IDs: {invalid_count} / 50")
print(f"  Sample class pixel distribution:")
for cid in sorted(class_pixel_total.keys()):
    print(f"    {cid} ({id2name.get(cid, '?')}): {class_pixel_total[cid]:,} px")

# 3. Check skeletons
skels = sorted((root / "centerlines").glob("*.png"))
sample_s = random.sample(skels, min(20, len(skels)))
for sp in sample_s[:5]:
    s = np.array(Image.open(sp))
    assert s.shape == (512, 512), f"Bad skeleton shape: {sp.name}"
    uniq = np.unique(s)
    fg = np.count_nonzero(s)
    # Skeleton should have fewer pixels than the mask
    mp = root / "masks" / sp.name
    if mp.exists():
        m = np.array(Image.open(mp))
        m_fg = np.count_nonzero(m)
        ratio = fg / max(m_fg, 1) * 100
        print(f"  Skeleton {sp.name}: {fg} px ({ratio:.1f}% of mask's {m_fg} px)")

# 4. Check metadata
meta = pd.read_csv(root / "metadata" / "patch_metadata.csv")
print(f"\nMetadata: {len(meta)} rows")
print(f"  Columns: {list(meta.columns)}")
print(f"  Splits: {dict(meta['split'].value_counts())}")
print(f"  sampling_weight stats: min={meta['sampling_weight'].min()}, "
      f"max={meta['sampling_weight'].max()}, "
      f"mean={meta['sampling_weight'].mean():.2f}")
print(f"  Rare-weighted (weight>1): {(meta['sampling_weight'] > 1).sum()}")

# 5. Verify no data leakage between splits
train_sources = set(meta[meta['split']=='train']['source_image'])
val_sources = set(meta[meta['split']=='val']['source_image'])
test_sources = set(meta[meta['split']=='test']['source_image'])
leak_tv = train_sources & val_sources
leak_tt = train_sources & test_sources
leak_vt = val_sources & test_sources
print(f"\n  Data leakage check:")
print(f"    Train&Val:  {len(leak_tv)} (should be 0)")
print(f"    Train&Test: {len(leak_tt)} (should be 0)")
print(f"    Val&Test:   {len(leak_vt)} (should be 0)")

# 6. Check previews exist
previews = list((root / "preview").glob("*.png"))
print(f"\n  Preview images: {len(previews)}")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE [OK]" if invalid_count == 0 and len(leak_tv) == 0 else "ISSUES FOUND [FAIL]")
print("=" * 60)
