import json
import cv2
import numpy as np
import glob
import os
from tqdm import tqdm

# --- CONFIGURATION ---
CLASSES_JSON = "./output/classes.json"   # Path to your classes.json
MASKS_DIR = r"C:\Users\Admin\line_detection\tiles\masks"      # Path to your masks (or ./tiles/masks)
# ---------------------

def check_consistency():
    # 1. Load Classes JSON
    if not os.path.exists(CLASSES_JSON):
        print(f"❌ Error: {CLASSES_JSON} not found!")
        return

    with open(CLASSES_JSON, 'r') as f:
        data = json.load(f)

    # Determine format and extract valid IDs
    # We expect format: {"Name": ID} (e.g. "Arrowhead": 1)
    # But we should handle {"ID": "Name"} just in case to warn the user.
    
    first_val = list(data.values())[0]
    
    if isinstance(first_val, int):
        print("ℹ️  JSON Format detected: {Name: ID} (Correct)")
        valid_ids = set(data.values())
        id_to_name = {v: k for k, v in data.items()}
    elif isinstance(first_val, str) and list(data.keys())[0].isdigit():
        print("⚠️  JSON Format detected: {ID: Name} (Inverted)")
        print("   (This works for some scripts but 'part2_prep.py' usually prefers Name:ID)")
        valid_ids = set([int(k) for k in data.keys()])
        id_to_name = {int(k): v for k, v in data.items()}
    else:
        print("❌ Error: Unknown JSON format.")
        return

    # Always add 0 (Background) if not present
    valid_ids.add(0)
    id_to_name[0] = "background"
    
    print(f"✅ Expecting these Class IDs: {sorted(list(valid_ids))}")

    # 2. Scan Masks
    mask_files = glob.glob(os.path.join(MASKS_DIR, "*.png"))
    if not mask_files:
        print(f"❌ No masks found in {MASKS_DIR}")
        return

    print(f"🔍 Scanning {len(mask_files)} masks...")
    
    global_unique = set()
    errors = []

    for f in tqdm(mask_files):
        mask = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"   ⚠️ Warning: Could not read {f}")
            continue
            
        unique = np.unique(mask)
        global_unique.update(unique)
        
        # Check for invalid values in this specific file
        diff = np.setdiff1d(unique, list(valid_ids))
        if len(diff) > 0:
            errors.append(f"{os.path.basename(f)} contains unknown IDs: {diff}")

    # 3. Report Results
    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)
    
    print(f"Found Pixel Values: {sorted(list(global_unique))}")
    print(f"Valid JSON IDs:     {sorted(list(valid_ids))}")
    
    # Check for Unknown IDs (The most dangerous error)
    unknown_ids = global_unique - valid_ids
    if unknown_ids:
        print(f"\n❌ CRITICAL ERROR: Found pixel values {unknown_ids} in masks that are NOT in classes.json!")
        print("   This will cause training to crash or produce garbage.")
        if len(errors) < 10:
            print("   Files with errors:\n   " + "\n   ".join(errors))
        else:
            print(f"   (Errors found in {len(errors)} files. First 5 shown:)\n   " + "\n   ".join(errors[:5]))
    else:
        print("\n✅ SUCCESS: All mask pixel values match your classes.json.")

    # Check for Missing Classes (Just a warning)
    missing_ids = valid_ids - global_unique
    if missing_ids:
        missing_names = [id_to_name.get(mid, 'Unknown') for mid in missing_ids]
        print(f"\n⚠️  Warning: The following classes are in JSON but NEVER appear in the masks:")
        print(f"   IDs: {missing_ids}")
        print(f"   Names: {missing_names}")
        print("   (This is fine if your dataset just happens to lack these objects, but check if you expected them.)")

if __name__ == "__main__":
    check_consistency()