import os
import sys
import json
import argparse
import glob
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- CONFIGURATION ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. EXACT NAME of the class you want to see (Must match classes.json exactly)
TARGET_CLASS = "Feature_Visible" 

# 2. COLOR to apply (BGR format: Blue, Green, Red) -> Currently Red
TARGET_COLOR = (0, 0, 255) 

# 3. SENSITIVITY (0.01 = 1% confidence). Increase this if you see too much noise.
SENSITIVITY_THRESHOLD = 0.01 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, required=True, help="Path to input image/folder")
    parser.add_argument('--model', type=str, required=True, help="Path to .pth file")
    parser.add_argument('--classes', type=str, default='./output/classes.json', help="Path to classes.json")
    parser.add_argument('--output', type=str, default='./test_results', help="Path to save results")
    parser.add_argument('--tile_size', type=int, default=1024)
    parser.add_argument('--overlap', type=int, default=256)
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
    # Try loading as SegFormer, fallback to DeepLab (matches your previous code)
    try:
        model = smp.SegFormer(encoder_name="mit_b2", classes=num_classes)
    except:
        model = smp.DeepLabV3Plus(encoder_name="resnet50", classes=num_classes)
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
                # Softmax gives probability (0.0 to 1.0)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                probs = probs.transpose(1, 2, 0)
            
            prob_buffer[y:y+tile_size, x:x+tile_size] += probs * window[..., None]
            count_buffer[y:y+tile_size, x:x+tile_size] += window

    with np.errstate(divide='ignore', invalid='ignore'):
        avg_prob = prob_buffer / count_buffer[..., None]
    return np.nan_to_num(avg_prob)[:h, :w, :]

def main():
    args = get_args()
    os.makedirs(args.output, exist_ok=True)
    
    # 1. Setup
    id_to_name = load_classes(args.classes)
    num_classes = len(id_to_name) + 1
    
    # Identify the Target ID
    target_id = None
    for cid, name in id_to_name.items():
        if name == TARGET_CLASS:
            target_id = cid
            break
            
    if target_id is None:
        print(f"\n[ERROR] '{TARGET_CLASS}' not found in {args.classes}")
        print(f"Available classes: {list(id_to_name.values())}")
        sys.exit(1)
        
    print(f"Targeting Class: '{TARGET_CLASS}' (ID: {target_id})")

    # 2. Model
    model = get_model(num_classes)
    model = safe_load_model(model, args.model)
    
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    files = glob.glob(os.path.join(args.images, "*.*")) if os.path.isdir(args.images) else [args.images]
    
    print(f"\nProcessing {len(files)} images with threshold {SENSITIVITY_THRESHOLD}...")

    for f in tqdm(files):
        img_name = Path(f).stem
        img = cv2.imread(f)
        if img is None: continue
        
        # Inference
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        probs = predict_large_image(model, img_rgb, args.tile_size, args.overlap, num_classes, transform)
        
        # --- DIAGNOSTIC STEP ---
        # Get the raw probability map for ONLY the target class
        target_prob_map = probs[:, :, target_id]
        
        # Check max confidence
        max_conf = np.max(target_prob_map)
        print(f"\nImage: {img_name} | Max Confidence for {TARGET_CLASS}: {max_conf:.4f}")
        
        if max_conf < SENSITIVITY_THRESHOLD:
            print(f"   WARNING: Signal too weak (< {SENSITIVITY_THRESHOLD}). Increasing exposure...")

        # --- VISUALIZATION 1: Heatmap (The Truth) ---
        # Normalize 0.0-1.0 to 0-255 for grayscale image
        heatmap_vis = (target_prob_map * 255).astype(np.uint8)
        # Apply colormap (Blue=Low, Red=High)
        heatmap_color = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(args.output, f"{img_name}_heatmap.jpg"), heatmap_color)

        # --- VISUALIZATION 2: The Overlay ---
        # Apply strict threshold
        mask_bool = (target_prob_map > SENSITIVITY_THRESHOLD)
        
        if np.any(mask_bool):
            # Thicken lines so they are visible
            mask_uint8 = mask_bool.astype(np.uint8)
            kernel = np.ones((3,3), np.uint8)
            mask_thick = cv2.dilate(mask_uint8, kernel, iterations=1).astype(bool)
            
            overlay = img.copy()
            color_layer = np.zeros_like(img, dtype=np.uint8)
            color_layer[mask_thick] = TARGET_COLOR
            
            # Heavy blending (keep lines bright)
            # 1.0 * Color + 0.5 * Original (Result > 255 is clipped, making lines very bright)
            overlay[mask_thick] = cv2.addWeighted(overlay[mask_thick], 0.4, color_layer[mask_thick], 0.9, 0)
            
            cv2.imwrite(os.path.join(args.output, f"{img_name}_result.jpg"), overlay)
            print(f"   -> Result Saved. {np.sum(mask_thick)} pixels detected.")
        else:
            print("   -> No pixels passed the threshold.")
            cv2.imwrite(os.path.join(args.output, f"{img_name}_result_EMPTY.jpg"), img)

    print(f"\nDone. Check '{args.output}' folder.")
    print("1. Look at *_heatmap.jpg -> If this is black, the model is not detecting anything.")
    print("2. Look at console -> If 'Max Confidence' is 0.0, check your class ID mapping.")

if __name__ == "__main__":
    main()