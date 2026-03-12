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

# Classes to IGNORE in visualization (Text, Background, etc.)
IGNORE_CLASSES = ["Arrowhead", "Dimension_text","Leader_line","Section_hatching_(cross-hatch)"]

# High contrast colors (BGR format for OpenCV)
COLORS_BGR = [
    (0, 0, 0),       # 0: Background (Ignored)
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
    parser = argparse.ArgumentParser(description="Test Engineering Drawing Line Detection")
    parser.add_argument('--images', type=str, required=True, help="Path to input image or folder")
    parser.add_argument('--model', type=str, required=True, help="Path to best_model.pth")
    parser.add_argument('--classes', type=str, default='./output/classes.json', help="Path to classes.json")
    parser.add_argument('--output', type=str, default='./test_results', help="Where to save results")
    parser.add_argument('--tile_size', type=int, default=1024, help="Inference tile size")
    parser.add_argument('--overlap', type=int, default=256, help="Overlap between tiles")
    return parser.parse_args()

def load_classes(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Handle {Name: ID} or {ID: Name} format
    first_val = list(data.values())[0]
    if isinstance(first_val, str) and list(data.keys())[0].isdigit():
        return {int(k): v for k, v in data.items()}
    else:
        return {v: k for k, v in data.items()}

def get_model(num_classes):
    try:
        model = smp.SegFormer(encoder_name="mit_b2", encoder_weights=None, in_channels=3, classes=num_classes)
    except:
        model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=num_classes)
    return model

def safe_load_model(model, path):
    print(f"Loading weights from {path}...")
    try:
        # Fix for PyTorch 2.6+ safe load issue
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

    # Sliding Window
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

def draw_legend(img, id_to_name):
    """Draws a legend box on the top-left, skipping ignored classes."""
    h, w = img.shape[:2]
    
    # Filter classes to show
    visible_items = {k: v for k, v in id_to_name.items() if v not in IGNORE_CLASSES}
    
    if not visible_items: return

    # Settings
    box_width = 280
    line_height = 30
    start_x, start_y = 10, 10
    padding = 10
    
    # Calculate box height
    num_items = len(visible_items)
    box_height = (num_items * line_height) + (padding * 2)
    
    # Draw semi-transparent background box
    overlay = img.copy()
    cv2.rectangle(overlay, (start_x, start_y), (start_x + box_width, start_y + box_height), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
    cv2.rectangle(img, (start_x, start_y), (start_x + box_width, start_y + box_height), (0, 0, 0), 1)
    
    # Draw items
    for i, (cls_id, name) in enumerate(visible_items.items()):
        color = COLORS_BGR[cls_id % len(COLORS_BGR)]
        y_pos = start_y + padding + (i * line_height) + 20
        
        # Color Swatch
        cv2.rectangle(img, (start_x + 10, y_pos - 15), (start_x + 30, y_pos + 5), color, -1)
        cv2.rectangle(img, (start_x + 10, y_pos - 15), (start_x + 30, y_pos + 5), (0,0,0), 1)
        
        # Text
        cv2.putText(img, name, (start_x + 40, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def draw_labels_on_lines(img, pred_mask, id_to_name):
    """Writes class labels on top of the detected lines."""
    for cls_id, name in id_to_name.items():
        if name in IGNORE_CLASSES: continue
        
        class_mask = (pred_mask == cls_id).astype(np.uint8)
        cnts, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in cnts:
            if cv2.contourArea(c) > 500: # Filter noise
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    # Draw text with white outline for readability
                    cv2.putText(img, name, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 4)
                    cv2.putText(img, name, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def main():
    args = get_args()
    os.makedirs(args.output, exist_ok=True)
    
    # 1. Setup
    id_to_name = load_classes(args.classes)
    num_classes = len(id_to_name) + 1
    print(f"Loaded {len(id_to_name)} classes.")
    print(f"Ignoring classes in visualization: {IGNORE_CLASSES}")

    # 2. Model
    model = get_model(num_classes)
    model = safe_load_model(model, args.model)
    
    # 3. Transform
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # 4. Images
    if os.path.isdir(args.images):
        files = glob.glob(os.path.join(args.images, "*.*"))
    else:
        files = [args.images]
    
    print(f"Processing {len(files)} images...")

    for f in tqdm(files):
        img_name = Path(f).stem
        img = cv2.imread(f)
        if img is None: continue
        
        # Convert BGR (OpenCV) to RGB (Model)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Inference
        probs = predict_large_image(model, img_rgb, args.tile_size, args.overlap, num_classes, transform)
        pred_mask = np.argmax(probs, axis=2).astype(np.uint8)
        
        # --- Visualization (Clean White Background) ---
        overlay = img.copy() # Start with original image
        
        # Loop through classes
        for cls_id in range(1, num_classes):
            name = id_to_name.get(cls_id, "Unknown")
            
            # SKIP HIDDEN CLASSES
            if name in IGNORE_CLASSES: 
                continue
            
            # Mask for this specific class
            mask_bool = (pred_mask == cls_id)
            
            if np.any(mask_bool):
                color = COLORS_BGR[cls_id % len(COLORS_BGR)]
                
                # Create colored layer
                colored_layer = np.zeros_like(img, dtype=np.uint8)
                colored_layer[mask_bool] = color
                
                # Extract original pixels at these locations
                roi = overlay[mask_bool]
                
                # Blend: 70% Color + 30% Original Image (High contrast)
                blended = cv2.addWeighted(roi, 0.3, colored_layer[mask_bool], 0.7, 0)
                
                # Apply blend back to overlay
                overlay[mask_bool] = blended

        # Draw Labels & Legend
        draw_labels_on_lines(overlay, pred_mask, id_to_name)
        draw_legend(overlay, id_to_name)
        
        # Save
        out_path = os.path.join(args.output, f"{img_name}_result.jpg")
        cv2.imwrite(out_path, overlay)

    print(f"\nDone! Results saved to {args.output}")

if __name__ == "__main__":
    main()