#!/usr/bin/env python3
"""
test_part1_infer.py

A compact, safe, CLI utility to run the Part-1 ignore/non-essential detector 
on one or many engineering drawings.

Modes:
  single: Process a single image.
  dir:    Process a directory of images (recursive).
  csv:    Process images listed in a CSV file.
  smoke:  Quick check on first few images of a dir.

Features:
  - Default Safe: Runs in dry-run mode (no changes to dataset) unless --apply is set.
  - Completeness: When --apply is set, images that don't need cleaning are copied 
    to the output folder to ensure the 'dataset_cleaned' is a complete mirror.
  - Robustness: Handles CPU/GPU selection, missing files, and inference errors.

Examples:
  # Preview single image (dry-run)
  python test_part1_infer.py --mode single --image_path test_images/x.png --weights_yolo models/yolov8_ignore.pt

  # Apply to folder (saves cleaned copies and completes dataset)
  python test_part1_infer.py --mode dir --input_dir dataset/Essential --weights_yolo models/yolov8_ignore.pt --apply --mask_mode blackout

  # Batch from CSV with GPU
  python test_part1_infer.py --mode csv --csv to_run.csv --weights_yolo models/yolov8_ignore.pt --device cuda --output_dir ./out_batch

  # Smoke test (quick check)
  python test_part1_infer.py --mode smoke --input_dir dataset/Essential --weights_yolo models/yolov8_ignore.pt
"""

import os
import sys
import argparse
import logging
import shutil
import time
import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm

# Optional imports with guards
try:
    import torch
    from ultralytics import YOLO
except ImportError:
    print("Error: torch and ultralytics are required. pip install torch ultralytics")
    sys.exit(1)

# Global Logger
logger = logging.getLogger("Part1Infer")

def setup_logging(output_dir, verbose=False):
    """Sets up console and file logging."""
    log_file = Path(output_dir) / "inference.log"
    level = logging.DEBUG if verbose else logging.INFO
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    logger.setLevel(level)
    if logger.hasHandlers(): logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def get_device(device_arg):
    """Safely determines the compute device."""
    if device_arg == 'cpu': return 'cpu'
    if device_arg == 'cuda' and torch.cuda.is_available(): return 0 
    return 'cpu'

class InferenceRunner:
    def __init__(self, args):
        self.args = args
        self.yolo_model = None
        self.seg_model = None
        self.device = get_device(args.device)
        self.load_models()

    def load_models(self):
        """Loads YOLO and optional segmentation models."""
        if not Path(self.args.weights_yolo).exists():
            logger.error(f"YOLO weights not found: {self.args.weights_yolo}")
            sys.exit(1)
        
        logger.info(f"Loading YOLO model: {self.args.weights_yolo} on {self.device}")
        self.yolo_model = YOLO(self.args.weights_yolo)
        
        if self.args.weights_seg and self.args.use_seg_refinement:
            try:
                import segmentation_models_pytorch as smp
                import torchvision.transforms as T
                self.T = T
                logger.info(f"Loading Seg model: {self.args.weights_seg}")
                self.seg_model = torch.load(self.args.weights_seg, map_location='cpu')
                # Handle state_dict vs full model
                if isinstance(self.seg_model, dict):
                    logger.warning("Loaded state_dict, instantiating default DeepLabV3+ ResNet50.")
                    model = smp.DeepLabV3Plus(encoder_name="resnet50", classes=2)
                    model.load_state_dict(self.seg_model)
                    self.seg_model = model
                
                device_str = 'cuda' if self.device == 0 else 'cpu'
                self.seg_model.to(device_str).eval()
            except Exception as e:
                logger.error(f"Failed to load Seg model: {e}")
                self.args.use_seg_refinement = False

    def predict_yolo(self, image):
        """Runs YOLOv8 inference and returns a combined binary mask."""
        results = self.yolo_model.predict(
            source=image, 
            save=False, 
            verbose=False, 
            device=self.device,
            retina_masks=True
        )
        
        h, w = image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        for result in results:
            if result.masks is None:
                continue
            for mask_tensor in result.masks.data:
                m = mask_tensor.cpu().numpy()
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h))
                combined_mask = cv2.bitwise_or(combined_mask, (m > 0.5).astype(np.uint8) * 255)
        
        return combined_mask

    def refine_seg(self, image):
        """Runs optional segmentation refinement."""
        if not self.seg_model: return None
        try:
            device_str = 'cuda' if self.device == 0 else 'cpu'
            img_t = self.T.ToTensor()(image).unsqueeze(0).to(device_str)
            with torch.no_grad():
                out = self.seg_model(img_t)
                if isinstance(out, torch.Tensor): pred = out
                else: pred = out['out']
                
                mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy().astype(np.uint8) * 255
                if mask.shape != image.shape[:2]:
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                return mask
        except Exception as e:
            logger.warning(f"Seg refinement failed: {e}")
            return None

    def process_single(self, image_path, output_root):
        """
        Processes a single image: Inference -> Stats -> Decision -> Output.
        Ensures 'dataset_cleaned' receives either the cleaned image OR a copy of the original.
        """
        res = {
            'image_path': str(image_path),
            'id': Path(image_path).stem,
            'processed_at': datetime.now().isoformat(),
            'error': ''
        }
        
        try:
            # 1. Load Image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError("Could not read image")
            h, w = img.shape[:2]
            
            # 2. Inference
            mask = self.predict_yolo(img)
            
            if self.args.use_seg_refinement:
                # Seg model usually expects RGB
                seg_mask = self.refine_seg(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if seg_mask is not None:
                    mask = cv2.bitwise_or(mask, seg_mask)

            # 3. Post-process (Morphological cleanup)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 4. Compute Stats & Decision
            mask_area_px = np.count_nonzero(mask)
            mask_fraction = mask_area_px / (h * w)
            
            should_apply = self.args.force_remove or (
                mask_fraction >= self.args.remove_threshold and 
                mask_area_px >= self.args.min_area_px
            )
            
            res.update({
                'mask_area_px': mask_area_px,
                'mask_fraction': round(mask_fraction, 5),
                'should_apply': should_apply
            })

            # 5. Determine relative path for output structure
            try:
                if self.args.mode in ['dir', 'smoke'] and self.args.input_dir:
                    rel_path = Path(image_path).relative_to(self.args.input_dir)
                else:
                    rel_path = Path(image_path).name
            except ValueError:
                rel_path = Path(image_path).name

            # 6. Define Output Paths
            mask_out = output_root / "final_masks" / f"{Path(image_path).stem}.png"
            overlay_out = output_root / "masked_overlays" / f"{Path(image_path).stem}_overlay.jpg"
            clean_out = output_root / "dataset_cleaned" / rel_path

            # Create directories
            mask_out.parent.mkdir(parents=True, exist_ok=True)
            overlay_out.parent.mkdir(parents=True, exist_ok=True)
            clean_out.parent.mkdir(parents=True, exist_ok=True)

            # 7. Write Debug Outputs (Always write for inspection)
            cv2.imwrite(str(mask_out), mask)
            res['mask_path'] = str(mask_out)

            # Write Overlay
            overlay = img.copy()
            overlay[mask > 0] = (0, 0, 255) # Red for removed areas
            weighted = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
            cv2.imwrite(str(overlay_out), weighted)
            res['overlay_path'] = str(overlay_out)

            # 8. Handle 'dataset_cleaned' (The Handover Logic)
            cleaned_path_str = ""
            if self.args.apply:
                if should_apply:
                    # Apply cleaning
                    if self.args.mask_mode == 'blackout':
                        cleaned = img.copy()
                        cleaned[mask > 0] = (255, 255, 255) # White background
                        cv2.imwrite(str(clean_out), cleaned)
                    elif self.args.mask_mode == 'save_mask':
                        # Just save the mask as the "cleaned" image? Rare usage.
                        cv2.imwrite(str(clean_out), mask)
                    else:
                        # Fallback
                        cleaned = img.copy()
                        cleaned[mask > 0] = (255, 255, 255)
                        cv2.imwrite(str(clean_out), cleaned)
                    res['action'] = 'cleaned'
                else:
                    # **CRITICAL**: Copy original if skipping cleaning
                    shutil.copy2(image_path, clean_out)
                    res['action'] = 'copied_original'
                
                cleaned_path_str = str(clean_out)
            else:
                res['action'] = 'dry_run_skipped'
            
            res['cleaned_path'] = cleaned_path_str

            # 9. Optional Evaluation
            if self.args.gt_masks_dir:
                gt_path = Path(self.args.gt_masks_dir) / f"{Path(image_path).stem}.png"
                if gt_path.exists():
                    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                    if gt is not None:
                        if gt.shape != mask.shape:
                            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
                        inter = np.logical_and(mask>0, gt>0).sum()
                        union = np.logical_or(mask>0, gt>0).sum()
                        iou = inter / (union + 1e-6)
                        res['iou'] = round(iou, 4)

        except Exception as e:
            res['error'] = str(e)
            logger.error(f"Error processing {image_path}: {e}")

        return res

def parse_args():
    parser = argparse.ArgumentParser(description="Part 1 Inference & Cleaning Utility")
    
    # Modes
    parser.add_argument("--mode", required=True, choices=['single', 'dir', 'csv', 'smoke'])
    parser.add_argument("--image_path", help="Path for single mode")
    parser.add_argument("--input_dir", help="Root dir for dir/smoke mode")
    parser.add_argument("--csv", help="CSV file for csv mode (must have 'image_path' column)")
    
    # Model
    parser.add_argument("--weights_yolo", required=True, help="Path to YOLOv8 ignore model")
    parser.add_argument("--weights_seg", help="Path to Seg refinement model")
    parser.add_argument("--use_seg_refinement", action="store_true")
    
    # Parameters
    parser.add_argument("--remove_threshold", type=float, default=0.02, help="Mask fraction to trigger cleaning")
    parser.add_argument("--min_area_px", type=int, default=50, help="Min pixels to trigger cleaning")
    parser.add_argument("--force_remove", action="store_true", help="Clean all images regardless of threshold")
    parser.add_argument("--mask_mode", default="blackout", choices=['blackout', 'save_mask', 'crop'])
    
    # Execution
    parser.add_argument("--apply", action="store_true", help="Apply changes (write cleaned dataset). Default is dry-run.")
    parser.add_argument("--output_dir", default="output_infer", help="Output root")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gt_masks_dir", help="Ground truth masks for evaluation")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup Dirs
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    setup_logging(out_root, args.verbose)
    
    logger.info(f"Starting Part 1 Inference in mode: {args.mode}")
    logger.info(f"Device: {args.device}, Apply: {args.apply}, Output: {args.output_dir}")

    # Gather Images
    images = []
    if args.mode == 'single':
        if not args.image_path: sys.exit("--image_path required for single mode")
        images = [Path(args.image_path)]
    elif args.mode == 'dir':
        if not args.input_dir: sys.exit("--input_dir required for dir mode")
        exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        images = [p for p in Path(args.input_dir).rglob("*") if p.suffix.lower() in exts]
    elif args.mode == 'csv':
        if not args.csv: sys.exit("--csv required for csv mode")
        import pandas as pd
        df = pd.read_csv(args.csv)
        if 'image_path' not in df.columns: sys.exit("CSV must have 'image_path' column")
        images = [Path(p) for p in df['image_path'].tolist()]
    elif args.mode == 'smoke':
        if not args.input_dir: sys.exit("--input_dir required for smoke mode")
        exts = {'.png', '.jpg', '.jpeg'}
        all_imgs = [p for p in Path(args.input_dir).rglob("*") if p.suffix.lower() in exts]
        images = all_imgs[:5] # Process first 5

    logger.info(f"Found {len(images)} images to process.")
    if not images:
        logger.warning("No images found. Exiting.")
        sys.exit(0)

    runner = InferenceRunner(args)
    results = []
    
    # Process
    # Use ThreadPool for concurrency. Note: YOLO object is not strictly thread-safe 
    # if sharing the same instance across threads without lock, but standard inference 
    # usually queues. For robust production, use ProcessPool or batched inference.
    # Here we use a simple loop if workers=1 (safer for GPU) or Pool for CPU.
    
    if args.num_workers <= 1:
        for img in tqdm(images, desc="Inferring"):
            results.append(runner.process_single(img, out_root))
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_img = {executor.submit(runner.process_single, img, out_root): img for img in images}
            for future in tqdm(as_completed(future_to_img), total=len(images), desc="Inferring"):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    logger.error(f"Worker error: {e}")

    # Save CSV
    if results:
        csv_path = out_root / "results.csv"
        keys = results[0].keys()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Results saved to {csv_path}")

    logger.info("Done.")

if __name__ == "__main__":
    main()