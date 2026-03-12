"""
part2_yolo.py

A single runnable Python script for training and running a YOLOv8-seg model 
specifically for small-object detection & instance segmentation (e.g., arrowheads, 
dimension text, datum symbols, tick marks).

Requirements:
    Python 3.8+
    pip install ultralytics opencv-python numpy pandas tqdm pyyaml

Usage:
    python part2_yolo.py prepare_labels --dataset_root ./tiles --classes_json classes.json
    python part2_yolo.py train --data yolo_data/data.yaml --epochs 100 --batch_size 8
    python part2_yolo.py infer --weights models/yolov8_core.pt --input_dir ./dataset/images --output ./output
    python part2_yolo.py eval --weights models/yolov8_core.pt --data yolo_data/data.yaml
"""

import argparse
import os
import sys
import json
import shutil
import random
import glob
import cv2
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm

# --- Dependency Check ---
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: 'ultralytics' library not found.")
    print("Run 'pip install ultralytics' to enable training and inference.")
    print("Only 'prepare_labels' mode will function correctly without it.")

# --- Constants & Config ---
DEFAULT_IMG_SIZE = 1280  # High res for small objects
DEFAULT_EPOCHS = 100
DEFAULT_BATCH = 16
DEFAULT_MODEL = "yolov8s-seg.pt"  # Pretrained start point (segmentation)
BEST_MODEL_PATH = os.path.join("models", "yolov8_core.pt")

# Target classes for this specific model
TARGET_CLASSES = [
   "Arrowhead",
   "Dimension_text",
]

# --- Helper Functions ---

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def normalize_polygon(polygon, width, height):
    """
    Normalizes polygon coordinates (x, y) to (x_n, y_n) in [0, 1].
    YOLO format expects: class x1 y1 x2 y2 ...
    """
    normalized = []
    for i in range(0, len(polygon), 2):
        x = polygon[i]
        y = polygon[i+1]
        normalized.append(x / width)
        normalized.append(y / height)
    return normalized

def mask_to_polygons(mask):
    """
    Converts a binary mask to a list of polygons.
    """
    # cv2.findContours expects uint8
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 10: # Filter very tiny noise
            poly = cnt.flatten().tolist()
            if len(poly) >= 6: # Need at least 3 points (x,y * 3)
                polygons.append(poly)
    return polygons

def save_yolo_label(txt_path, class_id, polygons, width, height):
    """
    Saves segmentation polygons to a text file in YOLO format.
    Row format: <class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
    """
    with open(txt_path, 'a') as f:
        for poly in polygons:
            norm_poly = normalize_polygon(poly, width, height)
            # Clip values to [0, 1] to prevent errors
            norm_poly = [min(max(x, 0.0), 1.0) for x in norm_poly]
            line = f"{class_id} " + " ".join([f"{c:.6f}" for c in norm_poly]) + "\n"
            f.write(line)

# --- CLI Command Implementations ---

def cmd_prepare_labels(args):
    """
    Converts dataset (images + per-class masks) to YOLOv8-seg format.
    Expected Input Structure:
      dataset_root/
         images/ (source images)
         masks/  (masks named {image_stem}_{classname}.png)
    
    Output Structure:
      output_dir/
         data.yaml
         train/images, train/labels
         val/images, val/labels
    """
    print(f"Preparing labels from {args.dataset_root}...")
    
    output_dir = args.output_dir
    ensure_dir(os.path.join(output_dir, 'train', 'images'))
    ensure_dir(os.path.join(output_dir, 'train', 'labels'))
    ensure_dir(os.path.join(output_dir, 'val', 'images'))
    ensure_dir(os.path.join(output_dir, 'val', 'labels'))

    # 1. Load class mapping
    if args.classes_json and os.path.exists(args.classes_json):
        with open(args.classes_json, 'r') as f:
            class_map_raw = json.load(f)
            if isinstance(class_map_raw, list):
                class_map = {name: i for i, name in enumerate(class_map_raw)}
            else:
                class_map = class_map_raw
    else:
        print(f"Classes JSON not found or not provided. Using built-in defaults: {TARGET_CLASSES}")
        class_map = {name: i for i, name in enumerate(TARGET_CLASSES)}

    # Filter to only the targets we care about for this model
    final_class_names = [name for name in class_map.keys() if name in TARGET_CLASSES]
    final_class_names.sort()
    
    # Create a dense ID map 0..N for YOLO
    yolo_id_map = {name: i for i, name in enumerate(final_class_names)}
    print(f"YOLO Class Mapping: {yolo_id_map}")

    # 2. Gather images
    source_images_dir = os.path.join(args.dataset_root, "images")
    source_masks_dir = os.path.join(args.dataset_root, "masks")
    
    if not os.path.exists(source_images_dir):
        print(f"Error: {source_images_dir} does not exist.")
        return

    all_images = glob.glob(os.path.join(source_images_dir, "*.*"))
    # Filter for image extensions
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    all_images = [f for f in all_images if f.lower().endswith(valid_exts)]
    
    if not all_images:
        print("No images found in dataset_root/images")
        return

    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.8) # 80/20 split
    train_imgs = all_images[:split_idx]
    val_imgs = all_images[split_idx:]
    
    datasets = [('train', train_imgs), ('val', val_imgs)]
    
    # 3. Process files
    for split, img_list in datasets:
        print(f"Processing {split} set ({len(img_list)} images)...")
        for img_path in tqdm(img_list):
            img_name = os.path.basename(img_path)
            img_stem = Path(img_path).stem
            
            # Read Image to get dims
            img = cv2.imread(img_path)
            if img is None: continue
            h, w = img.shape[:2]
            
            # Copy image to dest
            dest_img_path = os.path.join(output_dir, split, 'images', img_name)
            shutil.copy2(img_path, dest_img_path)
            
            # Create Label File
            label_txt_path = os.path.join(output_dir, split, 'labels', f"{img_stem}.txt")
            open(label_txt_path, 'w').close() # Create empty file
            
            # Find corresponding masks for target classes
            for cls_name in final_class_names:
                # Expect mask naming convention: filename_classname.png
                mask_name = f"{img_stem}_{cls_name}.png"
                mask_path = os.path.join(source_masks_dir, mask_name)
                
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        polys = mask_to_polygons(mask)
                        if polys:
                            yolo_id = yolo_id_map[cls_name]
                            save_yolo_label(label_txt_path, yolo_id, polys, w, h)
            
            # (Optional) Implement oversampling logic here:
            # If split == 'train' and we found small objects, we could crop 
            # sub-regions and save them as new training samples to output_dir.

    # 4. Generate data.yaml
    yaml_data = {
        'path': os.path.abspath(output_dir),
        'train': 'train/images',
        'val': 'val/images',
        'names': {i: name for i, name in enumerate(final_class_names)}
    }
    
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)
        
    print(f"Dataset preparation complete. YAML config saved to {yaml_path}")


def cmd_train(args):
    """Wrapper for Ultralytics YOLO training."""
    if not ULTRALYTICS_AVAILABLE:
        print("Error: Ultralytics not installed.")
        return

    print(f"Starting training with model {args.weights}...")
    
    model = YOLO(args.weights)

    # Train args
    # imgsz=1280 is highly recommended for small engineering symbols
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        project='yolo_runs',
        name='small_obj_run',
        exist_ok=True,
        save=True,
        patience=20,       # Early stopping
        mosaic=1.0,        # Mosaic helps with small objects context
        rect=False         # Rectangular training might be faster but square is safer for small obj
    )

    # Save best model to standard location
    ensure_dir(os.path.dirname(BEST_MODEL_PATH))
    
    # Attempt to locate best.pt
    # Standard path: yolo_runs/small_obj_run/weights/best.pt
    run_best_path = os.path.join('yolo_runs', 'small_obj_run', 'weights', 'best.pt')
    if os.path.exists(run_best_path):
        shutil.copy2(run_best_path, BEST_MODEL_PATH)
        print(f"Best model copied to {BEST_MODEL_PATH}")
    else:
        print(f"Warning: Could not auto-locate best.pt at {run_best_path}")
    
    print("Training complete.")


def cmd_infer(args):
    """Run inference on images, save masks and summary JSON."""
    if not ULTRALYTICS_AVAILABLE:
        print("Error: Ultralytics not installed.")
        return

    model_path = args.weights
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    model = YOLO(model_path)
    
    ensure_dir(args.output)
    output_inst_dir = os.path.join(args.output, "instances")
    ensure_dir(output_inst_dir)

    images = glob.glob(os.path.join(args.input_dir, "*.*"))
    print(f"Running inference on {len(images)} images...")

    summary_data = {}

    for img_path in tqdm(images):
        img_name = os.path.basename(img_path)
        
        # Inference with retina_masks=True for high-quality segmentation edges
        results = model.predict(
            source=img_path, 
            imgsz=args.imgsz, 
            conf=args.conf, 
            iou=args.iou, 
            save=False, 
            retina_masks=True,
            verbose=False
        )
        
        result = results[0]
        
        # Prepare output directory per image
        img_out_dir = os.path.join(output_inst_dir, Path(img_name).stem)
        ensure_dir(img_out_dir)

        # Save annotated overlay image
        res_plotted = result.plot()
        cv2.imwrite(os.path.join(img_out_dir, "annotated.jpg"), res_plotted)

        image_objects = []

        if result.masks is not None:
            # result.masks.data contains masks on the GPU usually, needs conversion
            # result.masks.xy contains list of polygon points (normalized=False)
            
            # To get binary masks efficiently:
            # We can rasterize the polygons or use the raw mask data if size matches
            h, w = result.orig_shape
            
            # Access boxes for class info
            boxes = result.boxes
            
            for i, poly_pts in enumerate(result.masks.xy):
                if len(poly_pts) == 0: continue

                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                class_name = result.names[cls_id]
                bbox = boxes.xyxy[i].tolist() # x1, y1, x2, y2

                # Create binary mask for this instance
                mask_img = np.zeros((h, w), dtype=np.uint8)
                int_poly = np.array(poly_pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask_img, [int_poly], 255)
                
                mask_filename = f"{class_name}_{i}_{conf:.2f}.png"
                cv2.imwrite(os.path.join(img_out_dir, mask_filename), mask_img)
                
                image_objects.append({
                    "class": class_name,
                    "confidence": conf,
                    "bbox": bbox,
                    "mask_file": mask_filename
                })

        summary_data[img_name] = image_objects

    # Save Summary
    with open(os.path.join(args.output, "inference_summary.json"), 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"Inference complete. Results in {args.output}")


def cmd_eval(args):
    """Evaluate model and save metrics."""
    if not ULTRALYTICS_AVAILABLE:
        print("Error: Ultralytics not installed.")
        return

    model = YOLO(args.weights)
    print("Evaluating model...")
    
    # Validate on 'val' split defined in data.yaml
    metrics = model.val(data=args.data, split='val', verbose=True)
    
    # Extract key metrics
    results = {
        "box_mAP50": metrics.box.map50,
        "box_mAP50-95": metrics.box.map,
        "seg_mAP50": metrics.seg.map50,
        "seg_mAP50-95": metrics.seg.map,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr
    }
    
    print("\nEvaluation Results:")
    print(json.dumps(results, indent=2))
    
    ensure_dir("output")
    out_file = os.path.join("output", "eval_instances.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to {out_file}")


# --- Main Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="Part 2: YOLOv8 Small Object Segmentation Pipeline")
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # prepare_labels
    p_prep = subparsers.add_parser('prepare_labels', help="Convert masks to YOLO format")
    p_prep.add_argument('--dataset_root', required=True, help="Root folder with 'images' and 'masks' subdirs")
    p_prep.add_argument('--output_dir', default="./yolo_data", help="Where to save YOLO dataset")
    p_prep.add_argument('--classes_json', default=None, help="JSON file with class list (optional)")

    # train
    p_train = subparsers.add_parser('train', help="Train YOLOv8 model")
    p_train.add_argument('--data', required=True, help="Path to data.yaml")
    p_train.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    p_train.add_argument('--batch_size', type=int, default=DEFAULT_BATCH)
    p_train.add_argument('--imgsz', type=int, default=DEFAULT_IMG_SIZE)
    p_train.add_argument('--weights', default=DEFAULT_MODEL)
    p_train.add_argument('--resume', action='store_true')

    # infer
    p_infer = subparsers.add_parser('infer', help="Run inference")
    p_infer.add_argument('--weights', required=True)
    p_infer.add_argument('--input_dir', required=True)
    p_infer.add_argument('--output', required=True)
    p_infer.add_argument('--imgsz', type=int, default=DEFAULT_IMG_SIZE)
    p_infer.add_argument('--conf', type=float, default=0.25)
    p_infer.add_argument('--iou', type=float, default=0.45)

    # eval
    p_eval = subparsers.add_parser('eval', help="Evaluate model")
    p_eval.add_argument('--weights', required=True)
    p_eval.add_argument('--data', required=True)

    args = parser.parse_args()

    if args.mode == 'prepare_labels':
        cmd_prepare_labels(args)
    elif args.mode == 'train':
        cmd_train(args)
    elif args.mode == 'infer':
        cmd_infer(args)
    elif args.mode == 'eval':
        cmd_eval(args)

if __name__ == "__main__":
    main()