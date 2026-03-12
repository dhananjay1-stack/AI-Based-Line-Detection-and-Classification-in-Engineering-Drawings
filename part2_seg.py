# part2_seg.py
#
# Requirements:
# pip install torch torchvision segmentation-models-pytorch timm albumentations opencv-python numpy tqdm pandas scikit-learn scikit-image
#
# Usage Examples:
#   Train:  python part2_seg.py train --dataset_root ./tiles --classes_json ./output/classes.json --out_dir ./output --epochs 50
#   Infer:  python part2_seg.py infer --weights ./output/models/best_model.pth --input_dir ./dataset/Essential/Wall/images --output ./output/inference
#   Eval:   python part2_seg.py eval --weights ./output/models/best_model.pth --gt_dir ./output/merged_masks --classes_json ./output/classes.json

import os
import sys
import json
import argparse
import logging
import random
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from PIL import Image

# --- Robust Imports ---
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.cuda.amp import autocast, GradScaler
    import segmentation_models_pytorch as smp
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from skimage.morphology import skeletonize, dilation, disk
    from sklearn.metrics import f1_score, jaccard_score
except ImportError as e:
    print(f"Error: Missing dependency '{e.name}'. Please install: pip install torch torchvision segmentation-models-pytorch timm albumentations opencv-python numpy tqdm pandas scikit-learn scikit-image")
    sys.exit(1)

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------------------------------------------------------
# Utils & Dataset
# -----------------------------------------------------------------------------

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_classes(json_path):
    if not os.path.exists(json_path):
        logger.error(f"Classes file not found: {json_path}")
        sys.exit(1)
    with open(json_path, 'r') as f:
        class_dict = json.load(f)
    # Sort by ID (0 is background, usually not in json, but we assume json is 1..N)
    # Returns list of (name, id) tuples
    return sorted(class_dict.items(), key=lambda x: x[1])

def get_class_weights(dataset, num_classes):
    """Calculate inverse frequency weights based on a subset of the training masks."""
    logger.info("Calculating class weights from training data subset...")
    counts = np.zeros(num_classes, dtype=np.float32)
    # Sample 10% or max 500 masks to estimate
    indices = np.random.choice(len(dataset), size=min(len(dataset), 500), replace=False)
    
    for i in indices:
        _, target = dataset[i]
        mask = target['mask'].numpy()
        unique, u_counts = np.unique(mask, return_counts=True)
        for u, c in zip(unique, u_counts):
            if u < num_classes:
                counts[u] += c
                
    # Inverse Frequency
    total = counts.sum() + 1e-6
    frequencies = counts / total
    weights = 1.0 / (frequencies + 0.01) # smoothed
    weights = weights / weights.max() # Normalize 0..1
    
    logger.info(f"Class Weights: {np.round(weights, 3)}")
    return torch.tensor(weights, dtype=torch.float32)

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, split_file=None, transform=None, use_boundary=False):
        self.root = Path(root_dir)
        self.img_dir = self.root / 'images'
        self.msk_dir = self.root / 'masks'
        self.transform = transform
        self.use_boundary = use_boundary
        
        # 1. Get all available tile files on disk
        # e.g., ['img1_0_0.png', 'img1_0_1024.png', 'img2_0_0.png']
        all_tiles = sorted([f.name for f in self.img_dir.glob('*.png')])
        
        # 2. Filter by split if provided
        if split_file and os.path.exists(split_file):
            with open(split_file, 'r') as f:
                # Read lines, strip whitespace, get the filename part only, remove extension
                # e.g., "C:/Users/.../img1.jpg" -> "img1"
                valid_stems = set(Path(line.strip()).stem for line in f.readlines())
            
            self.images = []
            for tile_name in all_tiles:
                # Tile format is: {original_stem}_{y}_{x}_{copy?}.png
                # We split by '_' and take the first part as the stem.
                # WARNING: If your original filenames have underscores (e.g. image_01.jpg),
                # splitting by '_' blindly breaks.
                # BETTER LOGIC: Remove the last 2-3 parts (coords) to find the stem.
                
                # Robust matching strategy:
                # Check if the tile filename STARTS with any valid stem
                # This is slower but safer for filenames with underscores.
                
                # Fast heuristic: 
                # "my_image_name_0_0.png" -> split('_') -> ['my', 'image', 'name', '0', '0.png']
                # The coordinates are always the last two numeric parts.
                
                stem_candidate = tile_name
                # Remove extension
                stem_candidate = os.path.splitext(stem_candidate)[0] 
                
                # Remove suffix _copyX if exists (oversampling)
                if "_copy" in stem_candidate:
                     stem_candidate = stem_candidate.rsplit('_copy', 1)[0]

                # Remove coordinates _Y_X
                parts = stem_candidate.rsplit('_', 2)
                if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
                    real_stem = parts[0]
                    # If the filename had underscores (e.g. "img_01"), rsplit might have cut too much 
                    # if the original name didn't end in numbers. 
                    # Actually, part2_prep saves tiles as f"{original_stem}_{y}_{x}"
                    # So rsplit('_', 2)[0] is exactly the original stem.
                    if real_stem in valid_stems:
                        self.images.append(tile_name)
        else:
            # If no split file found, use all images (fallback)
            self.images = all_tiles

        if len(self.images) == 0:
            logger.warning(f"No images found in {root_dir} matching split {split_file}")
            # Fallback debug: print what we saw
            if len(all_tiles) > 0:
                logger.info(f"Disk has {len(all_tiles)} tiles. Example: {all_tiles[0]}")
                if split_file:
                    logger.info(f"Split file loaded {len(valid_stems)} stems. Example: {list(valid_stems)[0]}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = str(self.img_dir / img_name)
        msk_path = str(self.msk_dir / img_name)
        
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
             # Fallback: sometimes mask filename might differ slightly? 
             # usually exact match for tiles.
             raise FileNotFoundError(f"Mask not found: {msk_path}")

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        target = {'mask': mask.long()}
        
        if self.use_boundary:
            binary = (mask.numpy() > 0).astype(np.uint8)
            if binary.max() > 0: # Avoid skeletonizing empty mask
                skel = skeletonize(binary).astype(np.uint8)
                bound = dilation(skel, disk(3))
                w_map = torch.from_numpy(bound).float() * 9.0 + 1.0
            else:
                w_map = torch.ones_like(target['mask']).float()
            target['weight'] = w_map

        return image, target

# -----------------------------------------------------------------------------
# Models & Metrics
# -----------------------------------------------------------------------------

def get_model(num_classes, architecture='SegFormer'):
    if architecture == 'SegFormer':
        try:
            model = smp.SegFormer(
                encoder_name="mit_b2", 
                encoder_weights="imagenet", 
                in_channels=3, 
                classes=num_classes
            )
            return model
        except Exception as e:
            logger.warning(f"SegFormer failed ({e}), falling back to DeepLabV3+")
            
    return smp.DeepLabV3Plus(
        encoder_name="resnet50", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=num_classes
    )

def calculate_metrics(pred_mask, gt_mask, num_classes):
    """Calc IoU per class for a batch."""
    # pred_mask: (H, W), gt_mask: (H, W)
    ious = []
    # Skip background (0) usually, but let's include it for completeness or skip
    # Requirements say per-class IoU.
    
    # We use macro average for mIoU
    for cls in range(num_classes):
        p = (pred_mask == cls)
        g = (gt_mask == cls)
        
        intersection = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        
        if union == 0:
            ious.append(np.nan) # Ignored in mean if class not present
        else:
            ious.append(intersection / union)
            
    return ious

# -----------------------------------------------------------------------------
# Modes
# -----------------------------------------------------------------------------

def run_train(args):
    seed_everything(args.seed)
    
    # 1. Setup Paths & Directories
    classes = load_classes(args.classes_json)
    num_classes = len(classes) + 1 # +1 for Background
    out_dir = Path(args.out_dir)
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Define Transforms
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    # 3. Splits
    split_dir = out_dir / "splits"
    train_split = split_dir / "train.txt"
    val_split = split_dir / "val.txt"
    
    if not train_split.exists():
        logger.warning(f"Split file {train_split} not found. Using all data.")
        train_split = None
        
    train_ds = SegmentationDataset(args.dataset_root, split_file=train_split, transform=train_transform, use_boundary=args.use_boundary_loss)
    val_ds = SegmentationDataset(args.dataset_root, split_file=val_split, transform=val_transform, use_boundary=False)
    
    if len(train_ds) == 0:
        logger.error("Training dataset is empty.")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 4. Initialize Model, Optimizer, Scaler
    model = get_model(num_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler()
    
    # 5. Resume & Early Stopping Configuration
    start_epoch = 0
    best_miou = 0.0
    patience = 11          # Stop if no improvement for 10 epochs
    no_improve_epochs = 0   # Counter
    
    if args.resume:
        chk_path = model_dir / "last_model.pth"
        if chk_path.exists():
            logger.info(f"Resuming training from {chk_path}...")
            try:
                checkpoint = torch.load(chk_path, map_location=DEVICE, weights_only=False)
            except TypeError:
                checkpoint = torch.load(chk_path, map_location=DEVICE)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                
                start_epoch = checkpoint['epoch'] + 1
                best_miou = checkpoint.get('best_miou', 0.0)
                no_improve_epochs = checkpoint.get('no_improve_epochs', 0)
                logger.info(f"Resumed successfully from Epoch {start_epoch+1}")
            else:
                model.load_state_dict(checkpoint)
                logger.warning("Legacy checkpoint found. Resetting state.")
        else:
            logger.warning(f"Resume requested but {chk_path} not found. Starting from scratch.")
    
    # 6. Class Weights & Loss
    cls_weights = get_class_weights(train_ds, num_classes).to(DEVICE)
    loss_fn_ce = nn.CrossEntropyLoss(weight=cls_weights, reduction='none')
    loss_fn_dice = smp.losses.DiceLoss(mode='multiclass')
    
    logger.info(f"Starting training loop from Epoch {start_epoch+1} to {args.epochs}...")
    
    # 7. Training Loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for img, target in pbar:
            img = img.to(DEVICE)
            mask = target['mask'].to(DEVICE)
            
            with autocast():
                preds = model(img)
                l_dice = loss_fn_dice(preds, mask)
                ce_pixel = loss_fn_ce(preds, mask)
                if 'weight' in target and args.use_boundary_loss:
                    w = target['weight'].to(DEVICE)
                    l_ce = (ce_pixel * w).mean()
                else:
                    l_ce = ce_pixel.mean()
                loss = l_dice + l_ce
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        # 8. Validation Loop
        model.eval()
        val_ious = []
        with torch.no_grad():
            for img, target in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                img = img.to(DEVICE)
                mask_gt = target['mask'].cpu().numpy()
                with autocast():
                    preds = model(img)
                    preds_mask = torch.argmax(preds, dim=1).cpu().numpy()
                for i in range(len(mask_gt)):
                    val_ious.append(calculate_metrics(preds_mask[i], mask_gt[i], num_classes))
        
        # Aggregate Metrics
        val_ious = np.array(val_ious)
        mean_iou_per_class = np.nanmean(val_ious, axis=0)
        miou = np.nanmean(mean_iou_per_class)
        
        # --- NEW: Display Per-Class Results ---
        # Map class indices back to names for display
        results_data = []
        
        # Index 0 is Background
        results_data.append({"Class": "Background", "IoU": mean_iou_per_class[0]})
        
        # Indices 1..N correspond to classes in json
        for name, cls_id in classes:
            cid = int(cls_id)
            if cid < len(mean_iou_per_class):
                results_data.append({"Class": name, "IoU": mean_iou_per_class[cid]})
        
        # Create a clean dataframe for printing
        df_res = pd.DataFrame(results_data)
        
        logger.info(f"\n{'='*20} Epoch {epoch+1} Results {'='*20}")
        logger.info(f"Train Loss: {train_loss/len(train_loader):.4f} | Val mIoU: {miou:.4f}")
        logger.info(f"Detailed IoU per Class:\n{df_res.to_string(index=False)}")
        logger.info("="*60)
        
        # 9. Checkpoint & Early Stopping Logic
        if miou > best_miou:
            best_miou = miou
            no_improve_epochs = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_miou': best_miou,
                'no_improve_epochs': 0
            }
            torch.save(checkpoint, model_dir / "best_model.pth")
            logger.info("  >>> New Best Model Saved!")
        else:
            no_improve_epochs += 1
            logger.info(f"  No improvement. Patience: {no_improve_epochs}/{patience}")
        
        # Save Last Model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_miou': best_miou,
            'no_improve_epochs': no_improve_epochs
        }
        torch.save(checkpoint, model_dir / "last_model.pth")
       
        # Stop Check
        if no_improve_epochs >= patience:
            logger.info(f"Early stopping triggered! No improvement for {patience} epochs.")
            break
        
def run_infer(args):
    # Load Model
    classes = load_classes(args.classes_json)
    num_classes = len(classes) + 1
    
    model = get_model(num_classes).to(DEVICE)
    
    # --- FIX: Safe Load & Dictionary Unwrapping ---
    try:
        checkpoint = torch.load(args.weights, map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.weights, map_location=DEVICE)
        
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)
    mask_out = output_dir / "semantic_masks"
    overlay_out = output_dir / "overlays"
    mask_out.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        overlay_out.mkdir(parents=True, exist_ok=True)
    
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    # Gaussian Window for fusion
    tile_size = args.tile_size
    overlap = args.overlap
    stride = tile_size - overlap
    
    def get_gaussian(size, sigma=0.5):
        x = np.linspace(-1, 1, size)
        x = np.exp(-0.5 * (x / sigma)**2)
        return np.outer(x, x)
        
    window = get_gaussian(tile_size)
    
    images = list(input_dir.glob("*.*"))
    logger.info(f"Found {len(images)} images to process.")
    
    for img_path in tqdm(images):
        if img_path.suffix.lower() not in ['.jpg', '.png', '.jpeg', '.tif', '.tiff']:
            continue
            
        # Read Image
        img_cv = cv2.imread(str(img_path))
        if img_cv is None: continue
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        try:
            prob_map = np.zeros((num_classes, h, w), dtype=np.float16) # Use float16 to save RAM
            count_map = np.zeros((h, w), dtype=np.float16)
        except MemoryError:
            logger.error(f"Image {img_path.name} is too large for RAM. Skipping.")
            continue
            
        # Sliding Window
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y2 = min(y + tile_size, h)
                x2 = min(x + tile_size, w)
                y1 = max(0, y2 - tile_size)
                x1 = max(0, x2 - tile_size)
                
                crop = img_rgb[y1:y2, x1:x2]
                
                # Transform
                tens = transform(image=crop)['image'].unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    logits = model(tens)
                    if args.tta:
                        logits_flip = model(torch.flip(tens, [3]))
                        logits += torch.flip(logits_flip, [3])
                        logits /= 2.0
                    
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0] # (C, H, W)
                
                # Accumulate
                prob_map[:, y1:y2, x1:x2] += probs * window[:y2-y1, :x2-x1]
                count_map[y1:y2, x1:x2] += window[:y2-y1, :x2-x1]
                
        # Average
        prob_map /= (count_map + 1e-6)
        
        # Save per class
        # Final Argmax for visualization
        final_mask = np.argmax(prob_map, axis=0).astype(np.uint8)
        
        # Save Binary Masks per class > threshold
        for idx, (cls_name, _) in enumerate(classes):
            cls_id = idx + 1 # 0 is bg
            cls_prob = prob_map[cls_id].astype(np.float32)
            
            binary = (cls_prob > args.prob_thresh).astype(np.uint8) * 255
            
            # Save if not empty
            if np.any(binary):
                cls_folder = mask_out / cls_name.replace(" ", "_")
                cls_folder.mkdir(exist_ok=True)
                cv2.imwrite(str(cls_folder / f"{img_path.stem}.png"), binary)
        
        # Visualize
        if args.visualize:
            vis_img = img_rgb.copy()
            # Generate random colors for classes
            colors = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
            colors[0] = [0,0,0]
            
            colored = colors[final_mask]
            overlay = cv2.addWeighted(vis_img, 0.6, colored, 0.4, 0)
            cv2.imwrite(str(overlay_out / f"{img_path.stem}_vis.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
def run_eval(args):
    logger.info("Starting Evaluation with Tiled Inference...")
    
    classes = load_classes(args.classes_json)
    num_classes = len(classes) + 1
    
    # Load model
    model = get_model(num_classes).to(DEVICE)
    try:
        checkpoint = torch.load(args.weights, map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.weights, map_location=DEVICE)
        
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    
    if not args.input_dir:
        logger.error("For Eval mode, please provide --input_dir pointing to original images.")
        sys.exit(1)
        
    input_path = Path(args.input_dir)
    gt_dir = Path(args.gt_dir)
    gt_files = list(gt_dir.glob("*.png"))
    
    metrics = {cls_name: {'iou': [], 'f1': []} for cls_name, _ in classes}
    
    # Tiling Config (Must match training/inference resolution)
    tile_size = args.tile_size
    overlap = args.overlap
    stride = tile_size - overlap
    
    # Gaussian Window for smoother fusion
    def get_gaussian(size, sigma=0.5):
        x = np.linspace(-1, 1, size)
        x = np.exp(-0.5 * (x / sigma)**2)
        return np.outer(x, x)
    window = get_gaussian(tile_size)

    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    logger.info(f"Evaluating {len(gt_files)} images...")

    for gt_file in tqdm(gt_files):
        # 1. Find Image
        img_candidates = list(input_path.glob(f"**/{gt_file.stem}.*"))
        img_file = next((f for f in img_candidates if f.suffix in ['.jpg','.png','.tif', '.jpeg']), None)
        
        if not img_file:
            continue
            
        # 2. Load GT & Image
        gt_mask = cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE)
        img_cv = cv2.imread(str(img_file))
        if img_cv is None: continue
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        h, w = img_rgb.shape[:2]
        
        # Resize GT if it doesn't match image (robustness)
        if gt_mask.shape != (h, w):
            gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
        # 3. Tiled Inference (The Correct Way)
        try:
            prob_map = np.zeros((num_classes, h, w), dtype=np.float16)
            count_map = np.zeros((h, w), dtype=np.float16)
        except MemoryError:
            logger.error(f"Image {img_file.name} too large for RAM. Skipping.")
            continue
            
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y2 = min(y + tile_size, h)
                x2 = min(x + tile_size, w)
                y1 = max(0, y2 - tile_size)
                x1 = max(0, x2 - tile_size)
                
                crop = img_rgb[y1:y2, x1:x2]
                
                tens = transform(image=crop)['image'].unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    logits = model(tens)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                
                # Accumulate with Gaussian weighting
                prob_map[:, y1:y2, x1:x2] += probs * window[:y2-y1, :x2-x1]
                count_map[y1:y2, x1:x2] += window[:y2-y1, :x2-x1]
        
        # Normalize & Argmax
        prob_map /= (count_map + 1e-6)
        pred_mask = np.argmax(prob_map, axis=0).astype(np.uint8)
            
        # 4. Calculate Metrics
        for i, (cls_name, cls_id) in enumerate(classes):
            idx = cls_id 
            
            p = (pred_mask == idx)
            g = (gt_mask == idx)
            
            intersect = np.logical_and(p, g).sum()
            union = np.logical_or(p, g).sum()
            
            if union > 0:
                iou = intersect / union
                metrics[cls_name]['iou'].append(iou)
                
                # Simple F1 (pixel-based)
                prec = intersect / (p.sum() + 1e-6)
                rec = intersect / (g.sum() + 1e-6)
                f1 = 2 * (prec * rec) / (prec + rec + 1e-6)
                metrics[cls_name]['f1'].append(f1)

    # 5. Report
    summary = {}
    print("\n" + "="*45)
    print(f"{'Class':<25} | {'mIoU':<10} | {'F1':<10}")
    print("-" * 45)
    for cls_name, data in metrics.items():
        m_iou = np.mean(data['iou']) if data['iou'] else 0.0
        m_f1 = np.mean(data['f1']) if data['f1'] else 0.0
        summary[cls_name] = {"mIoU": m_iou, "F1": m_f1}
        print(f"{cls_name:<25} | {m_iou:.4f}     | {m_f1:.4f}")
        
    avg_miou = np.mean([v['mIoU'] for v in summary.values()])
    print("-" * 45)
    print(f"{'MEAN':<25} | {avg_miou:.4f}     |")
    print("="*45)
    
    ensure_dir = Path(args.output)
    ensure_dir.mkdir(parents=True, exist_ok=True)
    with open(ensure_dir/"eval_seg.json", "w") as f:
        json.dump(summary, f, indent=4)
def get_args():
    parser = argparse.ArgumentParser(description="Part-2 Segmentation")
    subparsers = parser.add_subparsers(dest='mode', required=True)
    
    # Parent args
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument('--dataset_root', type=str, default='./tiles')
    parent.add_argument('--classes_json', type=str, default='./output/classes.json')
    parent.add_argument('--seed', type=int, default=42)
    parent.add_argument('--device', type=str, default='cuda')
    
    # Train
    p_train = subparsers.add_parser('train', parents=[parent])
    p_train.add_argument('--resume', action='store_true', help="Resume training from last checkpoint")
    p_train.add_argument('--out_dir', type=str, default='./output')
    p_train.add_argument('--epochs', type=int, default=50)
    p_train.add_argument('--batch_size', type=int, default=8)
    p_train.add_argument('--lr', type=float, default=1e-4)
    p_train.add_argument('--use_boundary_loss', action='store_true')
    
    # Infer
    p_infer = subparsers.add_parser('infer', parents=[parent])
    p_infer.add_argument('--weights', type=str, required=True)
    p_infer.add_argument('--input_dir', type=str, required=True)
    p_infer.add_argument('--output', type=str, default='./output')
    p_infer.add_argument('--tile_size', type=int, default=1024)
    p_infer.add_argument('--overlap', type=int, default=128)
    p_infer.add_argument('--prob_thresh', type=float, default=0.5)
    p_infer.add_argument('--tta', action='store_true')
    p_infer.add_argument('--visualize', action='store_true')
    
    # Eval
    p_eval = subparsers.add_parser('eval', parents=[parent])
    p_eval.add_argument('--weights', type=str, required=True)
    p_eval.add_argument('--gt_dir', type=str, required=True)
    p_eval.add_argument('--input_dir', type=str, help="Path to original images for Eval")
    p_eval.add_argument('--output', type=str, default='./output')
    p_eval.add_argument('--tile_size', type=int, default=1024)
    p_eval.add_argument('--overlap', type=int, default=128)
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    if args.mode == 'train':
        run_train(args)
    elif args.mode == 'infer':
        run_infer(args)
    elif args.mode == 'eval':
        run_eval(args)