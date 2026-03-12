import os
import sys
import json
import argparse
import glob
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- CONFIGURATION ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Classes to HIDE in the final image (but we will still check if they exist in the mask)
IGNORE_CLASSES_VISUAL = ["Arrowhead", "Dimension_text", "Leader_line", "Section_hatching_(cross-hatch)"]

# High contrast colors (BGR format for OpenCV)
COLORS_BGR = [
    (0, 0, 0),       # 0: Background
    (0, 0, 255),     # 1: Red
    (0, 255, 0),     # 2: Green
    (255, 0, 0),     # 3: Blue
    (0, 255, 255),   # 4: Yellow
    (255, 0, 255),   # 5: Magenta
    (255, 255, 0),   # 6: Cyan
    (255, 128, 0),   # 7: Blue-ish
    (128, 0, 255),   # 8: Purple
    (0, 128, 255),   # 9: Orange
    (128, 255, 0),   # 10: Lime
    (255, 128, 128), # 11: Pink
    (128, 128, 255), # 12: Light Red
]

def get_args():
    parser = argparse.ArgumentParser(description="Audit Engineering Drawing Line Detection")
    parser.add_argument('--images', type=str, required=True, help="Path to input image or folder")
    parser.add_argument('--model', type=str, required=True, help="Path to best_model.pth")
    parser.add_argument('--classes', type=str, default='./output/classes.json', help="Path to classes.json")
    parser.add_argument('--output', type=str, default='./audit_results', help="Where to save results")
    parser.add_argument('--tile_size', type=int, default=1024, help="Inference tile size")
    parser.add_argument('--overlap', type=int, default=256, help="Overlap between tiles")
    return parser.parse_args()

def load_classes(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    first_val = list(data.values())[0]
    if isinstance(first_val, str) and list(data.keys())[0].isdigit():
        return {int(k): v for k, v in data.items()}
    else:
        return {v: k for k, v in data.items()}

def get_model(num_classes):
    # Attempt to load SegFormer, fallback to DeepLab (adjust as per your training)
    try:
        model = smp.SegFormer(encoder_name="mit_b2", encoder_weights=None, in_channels=3, classes=num_classes)
    except:
        model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=num_classes)
    return model

def safe_load_model(model, path):
    print(f"Loading weights from {path}...")
    try:
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model

def get_gaussian_window(size, sigma=0.5):
    x = np.linspace(-1, 1, size)
    x = np.exp(-0.5 * (x / sigma)**2)
    return np.outer(x, x)

def predict_large_image(model, img, tile_size, overlap, num_classes, transform):
    h, w = img.shape[:2]
    stride = tile_size - overlap
    pad_h = (tile_size - (h % stride)) % stride
    pad_w = (tile_size - (w % stride)) % stride
    if h < tile_size: pad_h += (tile_size - h)
    if w < tile_size: pad_w += (tile_size - w)
    
    img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    h_pad, w_pad = img_padded.shape[:2]
    
    prob_buffer = np.zeros((h_pad, w_pad, num_classes), dtype=np.float32)
    count_buffer = np.zeros((h_pad, w_pad), dtype=np.float32)
    window = get_gaussian_window(tile_size)

    for y in range(0, h_pad - tile_size + 1, stride):
        for x in range(0, w_pad - tile_size + 1, stride):
            tile = img_padded[y:y+tile_size, x:x+tile_size]
            aug = transform(image=tile)["image"]
            tensor = aug.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                probs = probs.transpose(1, 2, 0)
            prob_buffer[y:y+tile_size, x:x+tile_size] += probs * window[..., None]
            count_buffer[y:y+tile_size, x:x+tile_size] += window

    with np.errstate(divide='ignore', invalid='ignore'):
        avg_prob = prob_buffer / count_buffer[..., None]
    return np.nan_to_num(avg_prob)[:h, :w, :]

def generate_report(pred_mask, id_to_name, total_pixels):
    """
    Analyzes the mask and returns a structured list of detected lines.
    """
    detected_stats = []
    
    # Get unique classes present in the mask
    unique_ids, counts = np.unique(pred_mask, return_counts=True)
    pixel_counts = dict(zip(unique_ids, counts))
    
    print("\n" + "="*80)
    print(f"{'ID':<4} | {'Class Name':<30} | {'Pixels':<10} | {'Status':<15} | {'Visualized?':<12} | {'Color (BGR)'}")
    print("-" * 80)
    
    for cls_id in range(1, len(id_to_name) + 1):
        name = id_to_name.get(cls_id, "Unknown")
        count = pixel_counts.get(cls_id, 0)
        color = COLORS_BGR[cls_id % len(COLORS_BGR)]
        
        # Determine Status
        if count == 0:
            status = "NOT FOUND"
        elif count < 500: # Threshold for noise
            status = "TRACE/NOISE"
        else:
            status = "DETECTED"
            
        # Determine Visualization Status
        is_ignored = name in IGNORE_CLASSES_VISUAL
        vis_status = "NO (Ignored)" if is_ignored else "YES"
        
        print(f"{cls_id:<4} | {name:<30} | {count:<10} | {status:<15} | {vis_status:<12} | {color}")
        
        detected_stats.append({
            "id": cls_id,
            "name": name,
            "pixels": count,
            "status": status,
            "color": color
        })
    print("="*80 + "\n")
    return detected_stats

def main():
    args = get_args()
    os.makedirs(args.output, exist_ok=True)
    
    # 1. Setup
    id_to_name = load_classes(args.classes)
    num_classes = len(id_to_name) + 1
    
    # 2. Model
    model = get_model(num_classes)
    model = safe_load_model(model, args.model)
    
    # 3. Transform
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # 4. Process Images
    if os.path.isdir(args.images):
        files = glob.glob(os.path.join(args.images, "*.*"))
    else:
        files = [args.images]
    
    print(f"Processing {len(files)} images...")

    for f in tqdm(files):
        img_name = Path(f).stem
        print(f"\nAnalyzing: {img_name}")
        
        img = cv2.imread(f)
        if img is None: continue
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Inference
        probs = predict_large_image(model, img_rgb, args.tile_size, args.overlap, num_classes, transform)
        pred_mask = np.argmax(probs, axis=2).astype(np.uint8)
        
        # --- GENERATE TEXT REPORT ---
        generate_report(pred_mask, id_to_name, h*w)
        
        # --- VISUALIZATION 1: OVERLAY (Standard) ---
        overlay = img.copy()
        
        # --- VISUALIZATION 2: MASK ONLY (Black Background - High Contrast) ---
        mask_vis = np.zeros_like(img) # Solid black
        
        for cls_id in range(1, num_classes):
            name = id_to_name.get(cls_id, "Unknown")
            mask_bool = (pred_mask == cls_id)
            
            if not np.any(mask_bool): continue
            
            color = COLORS_BGR[cls_id % len(COLORS_BGR)]
            
            # 1. Update Mask Only Image (Draw ALL classes, even ignored ones, to prove detection)
            mask_vis[mask_bool] = color
            
            # 2. Update Overlay (Skip ignored classes)
            if name not in IGNORE_CLASSES_VISUAL:
                colored_layer = np.zeros_like(img, dtype=np.uint8)
                colored_layer[mask_bool] = color
                roi = overlay[mask_bool]
                blended = cv2.addWeighted(roi, 0.3, colored_layer[mask_bool], 0.7, 0)
                overlay[mask_bool] = blended

        # Save Visuals
        cv2.imwrite(os.path.join(args.output, f"{img_name}_overlay.jpg"), overlay)
        cv2.imwrite(os.path.join(args.output, f"{img_name}_mask_only.jpg"), mask_vis)

    print(f"\nDone! Check '{args.output}' for images and scroll up for the text report.")

if __name__ == "__main__":
    main()