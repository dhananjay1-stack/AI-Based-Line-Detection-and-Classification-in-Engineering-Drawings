import cv2
import numpy as np
import glob
import os

# 1. Find a mask tile
tile_dir = "./tiles/masks"
files = glob.glob(os.path.join(tile_dir, "*.png"))

if not files:
    print("No tiles found to check!")
else:
    # Pick the first file
    mask_path = files[0]
    print(f"Checking file: {mask_path}")

    # 2. Load it carefully (Don't change values)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 3. Check the values
    unique_values = np.unique(mask)
    print(f"Unique pixel values found: {unique_values}")

    if len(unique_values) > 1:
        print("✅ SUCCESS: The mask contains hidden classes!")
        print(f"   (It has background {unique_values[0]} and classes {unique_values[1:]})")
        
        # 4. Save a VISIBLE version just for you to see
        visible_mask = mask * 20  # Multiply IDs by 20 to make them bright
        cv2.imwrite("visible_debug_mask.png", visible_mask)
        print("   Saved 'visible_debug_mask.png' - open this to see the lines.")
    else:
        print("❌ FAILURE: This mask is actually empty (all zeros).")