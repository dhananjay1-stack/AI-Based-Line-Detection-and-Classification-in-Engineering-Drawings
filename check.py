import cv2
import numpy as np
import glob
import json

# Load class map
with open("./output/classes.json", "r") as f:
    class_map = json.load(f)
    print(f"Class Map: {class_map}")

# IDs for Arrowhead (1) and Dimension_text (6) based on your uploaded json
TARGET_IDS = [1, 6] 

print("Scanning for Arrowheads (1) and Text (6)...")
files = glob.glob("./tiles/masks/*.png")
found = False

for f in files[:500]: # Check first 500 tiles
    mask = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    unique = np.unique(mask)
    
    # Check if any target ID is in this mask
    if np.any(np.isin(unique, TARGET_IDS)):
        print(f"✅ Found Target in {f}: IDs {unique}")
        found = True
        break

if not found:
    print("❌ PROBLEM: No Arrowheads (ID 1) or Text (ID 6) found in the first 500 tiles.")
    print("   Your YOLO labels are empty because the data is missing from the tiles.")