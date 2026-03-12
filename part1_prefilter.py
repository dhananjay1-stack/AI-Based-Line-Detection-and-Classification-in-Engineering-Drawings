#!/usr/bin/env python3
"""
part1_prefilter.py

Engineering Drawing Dataset Pre-filter Pipeline
Validates, normalizes, converts, and filters engineering drawing datasets.

Usage examples:
  python part1_prefilter.py --mode stats --dataset_root ./dataset --preview_n 20
  python part1_prefilter.py --mode convert --dataset_root ./dataset --convert_format yolo
  python part1_prefilter.py --mode train --train_yolov8 --epochs 50 --batch 8
  python part1_prefilter.py --mode infer --weights_yolo modelkls/yolov8_ignore.pt --mask_mode blackout --dry_run
  python part1_prefilter.py --mode full_pipeline --train_yolov8
"""

# pip install numpy opencv-python Pillow tqdm pandas ultralytics torch torchvision segmentation-models-pytorch timm scikit-image shapely matplotlib

import os
import sys
import argparse
import json
import csv
import random
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import traceback
import ast

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not installed. Some stats features may be limited.")
    print("Install with: pip install pandas")

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False

try:
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class Config:
    mode: str = "help"
    dataset_root: str = "./dataset"
    output_dir: str = "./output"
    dataset_normalized: str = "./dataset_normalized"
    dataset_yolo: str = "./dataset_yolo"
    dataset_coco: str = "./dataset_coco"
    splits_dir: str = "./splits"
    models_dir: str = "./models"
    weights_yolo: Optional[str] = None
    weights_seg: Optional[str] = None
    train_yolov8: bool = False
    train_seg: bool = False
    convert_format: str = "yolo"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42
    remove_threshold: float = 0.02
    min_area_px: int = 50
    preserve_thin_classes: str = "Center_line,hidden_lines,Phantom_lines,Feature_Visible,Section hatching (cross-hatch)"
    mask_mode: str = "blackout"
    dry_run: bool = False
    preview_n: int = 10
    verbose: bool = False
    num_workers: int = 4
    config: Optional[str] = None
    ignore_classes: Optional[str] = None
    force_remove: bool = False
    use_seg_refinement: bool = False
    poly_epsilon: float = 2.0
    imgsz: int = 640
    epochs: int = 50
    batch: int = 6
    lr: float = 0.001


def find_contours_safe(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE):
    result = cv2.findContours(mask, mode, method)
    if len(result) == 3:
        _, contours, hierarchy = result
    else:
        contours, hierarchy = result
    return contours, hierarchy


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_class_name(class_name: str) -> str:
    return class_name.replace(" ", "_")


def load_config_from_yaml(yaml_path: str) -> Dict:
    if not HAS_YAML:
        print("Error: PyYAML not installed. Cannot load config file.")
        print("Install with: pip install pyyaml")
        sys.exit(1)
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


class DatasetValidator:
    def __init__(self, config: Config):
        self.config = config
        self.stats = []
        self.errors = []
        self.class_folders = []
        
    def discover_classes(self) -> List[Tuple[str, str, str]]:
        classes = []
        root = Path(self.config.dataset_root)
        
        for category in ['Essential', 'Non_essential']:
            category_path = root / category
            if not category_path.exists():
                continue
                
            for class_folder in category_path.iterdir():
                if not class_folder.is_dir():
                    continue
                    
                images_path = class_folder / 'images'
                masks_path = class_folder / 'masks'
                
                if images_path.exists() and masks_path.exists():
                    classes.append((category, class_folder.name, str(class_folder)))
                    
        return classes
    
    def validate_and_compute_stats(self):
        classes = self.discover_classes()
        self.class_folders = classes
        
        print(f"\nFound {len(classes)} class folders")
        
        for category, class_name, class_path in tqdm(classes, desc="Validating classes"):
            self._process_class(category, class_name, class_path)
        
        self._save_stats()
        self._generate_summary()
        
    def _process_class(self, category: str, class_name: str, class_path: str):
        images_path = Path(class_path) / 'images'
        masks_path = Path(class_path) / 'masks'
        
        image_files = list(images_path.glob('*'))
        valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        image_files = [f for f in image_files if f.suffix.lower() in valid_exts]
        
        for img_file in image_files:
            try:
                self._process_image_mask_pair(category, class_name, img_file, masks_path)
            except Exception as e:
                error_msg = f"Error processing {img_file}: {str(e)}"
                self.errors.append(error_msg)
                if self.config.verbose:
                    print(f"\n{error_msg}")
                    traceback.print_exc()
        
    def _process_image_mask_pair(self, category: str, class_name: str, img_file: Path, masks_path: Path):
        mask_candidates = [
            masks_path / f"{img_file.stem}.png",
            masks_path / f"{img_file.stem}.jpg",
            masks_path / f"{img_file.stem}.jpeg",
            masks_path / f"{img_file.stem}.tif",
            masks_path / f"{img_file.stem}.tiff",
        ]
        
        mask_file = None
        for candidate in mask_candidates:
            if candidate.exists():
                mask_file = candidate
                break
        
        if mask_file is None:
            self.errors.append(f"Missing mask for {img_file}")
            return
        
        image = cv2.imread(str(img_file))
        if image is None:
            self.errors.append(f"Failed to read image: {img_file}")
            return
            
        h, w = image.shape[:2]
        
        mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
        if mask is None:
            self.errors.append(f"Failed to read mask: {mask_file}")
            return
        
        # --- FIX STARTS HERE ---
        # Handle various channel configurations safely
        if len(mask.shape) == 3:
            if mask.shape[2] == 3:  # BGR -> Gray
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            elif mask.shape[2] == 4:  # BGRA -> Gray
                mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
            elif mask.shape[2] == 1:  # (H, W, 1) -> (H, W)
                mask = mask[:, :, 0]
        # --- FIX ENDS HERE ---
        
        elif len(mask.shape) == 2:
            pass
        else:
            self.errors.append(f"Invalid mask shape: {mask.shape} for {mask_file}")
            return
        
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        mask_area_px = np.sum(mask_binary > 0)
        mask_fraction = mask_area_px / (w * h) if (w * h) > 0 else 0
        
        contours, _ = find_contours_safe(mask_binary)
        
        bbox = [0, 0, 0, 0]
        if contours:
            all_points = np.vstack(contours)
            x, y, bw, bh = cv2.boundingRect(all_points)
            bbox = [x, y, x + bw, y + bh]
        
        self.stats.append({
            'image_id': img_file.stem,
            'category': category,
            'class': class_name,
            'image_path': str(img_file),
            'mask_path': str(mask_file),
            'image_width': w,
            'image_height': h,
            'mask_area_px': int(mask_area_px),
            'mask_bbox': bbox,
            'mask_fraction': float(mask_fraction)
        })
        
    def _save_stats(self):
        ensure_dir(self.config.output_dir)
        stats_path = Path(self.config.output_dir) / 'stats.csv'
        
        if HAS_PANDAS:
            df = pd.DataFrame(self.stats)
            df.to_csv(stats_path, index=False)
        else:
            if self.stats:
                with open(stats_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.stats[0].keys())
                    writer.writeheader()
                    writer.writerows(self.stats)
        
        log_path = Path(self.config.output_dir) / 'log.txt'
        with open(log_path, 'w') as f:
            f.write("Validation Errors:\n")
            for error in self.errors:
                f.write(f"{error}\n")
        
        print(f"\nStats saved to: {stats_path}")
        print(f"Errors logged to: {log_path}")
        print(f"Total images processed: {len(self.stats)}")
        print(f"Total errors: {len(self.errors)}")
    
    def _generate_summary(self):
        if not self.stats:
            return
            
        summary_path = Path(self.config.output_dir) / 'class_summary.csv'
        
        class_stats = defaultdict(lambda: {'count': 0, 'total_area': 0, 'total_fraction': 0})
        
        for stat in self.stats:
            key = stat['class']
            class_stats[key]['count'] += 1
            class_stats[key]['total_area'] += stat['mask_area_px']
            class_stats[key]['total_fraction'] += stat['mask_fraction']
        
        summary = []
        for class_name, stats in class_stats.items():
            summary.append({
                'class': class_name,
                'image_count': stats['count'],
                'avg_mask_area_px': stats['total_area'] / stats['count'],
                'avg_mask_fraction': stats['total_fraction'] / stats['count']
            })
        
        if HAS_PANDAS:
            df = pd.DataFrame(summary)
            df.to_csv(summary_path, index=False)
        else:
            with open(summary_path, 'w', newline='') as f:
                if summary:
                    writer = csv.DictWriter(f, fieldnames=summary[0].keys())
                    writer.writeheader()
                    writer.writerows(summary)
        
        print(f"Class summary saved to: {summary_path}")
    
    def load_stats_from_csv(self) -> bool:
        stats_path = Path(self.config.output_dir) / 'stats.csv'
        if not stats_path.exists():
            return False
            
        print(f"Loading cached stats from {stats_path}...")
        try:
            if HAS_PANDAS:
                df = pd.read_csv(stats_path)
                # Convert string "[0,0,10,10]" back to list [0,0,10,10]
                df['mask_bbox'] = df['mask_bbox'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                self.stats = df.to_dict('records')
            else:
                with open(stats_path, 'r') as f:
                    reader = csv.DictReader(f)
                    self.stats = []
                    for row in reader:
                        # Manual type conversion required for CSV
                        row['mask_area_px'] = int(row['mask_area_px'])
                        row['mask_fraction'] = float(row['mask_fraction'])
                        row['image_width'] = int(row['image_width'])
                        row['image_height'] = int(row['image_height'])
                        row['mask_bbox'] = ast.literal_eval(row['mask_bbox'])
                        self.stats.append(row)
            
            print(f"Loaded {len(self.stats)} images from cache.")
            return True
        except Exception as e:
            print(f"Warning: Failed to load cached stats ({e}). Re-computing...")
            return False

class DatasetNormalizer:
    def __init__(self, config: Config, validator: DatasetValidator):
        self.config = config
        self.validator = validator
        
    def normalize_masks(self):
        print("\nNormalizing masks...")
        ensure_dir(self.config.dataset_normalized)
        
        for stat in tqdm(self.validator.stats, desc="Normalizing"):
            try:
                self._normalize_mask(stat)
            except Exception as e:
                if self.config.verbose:
                    print(f"\nError normalizing {stat['mask_path']}: {e}")
                    traceback.print_exc()
    
    def _normalize_mask(self, stat: Dict):
        mask = cv2.imread(stat['mask_path'], cv2.IMREAD_UNCHANGED)
        if mask is None:
            return
        
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        relative_path = Path(stat['mask_path']).relative_to(self.config.dataset_root)
        output_path = Path(self.config.dataset_normalized) / relative_path
        
        ensure_dir(str(output_path.parent))
        cv2.imwrite(str(output_path), mask_binary)


class DatasetConverter:
    def __init__(self, config: Config, validator: DatasetValidator):
        self.config = config
        self.validator = validator
        self.class_to_id = {}
        self.id_to_class = {}
        
    def convert(self):
        self._build_class_mapping()
        self._create_splits()
        
        if self.config.convert_format == 'yolo':
            self._convert_to_yolo()
        elif self.config.convert_format == 'coco':
            self._convert_to_coco()
        else:
            print(f"Unknown conversion format: {self.config.convert_format}")
    
    def _build_class_mapping(self):
        unique_classes = sorted(set(stat['class'] for stat in self.validator.stats))
        self.class_to_id = {cls: idx for idx, cls in enumerate(unique_classes)}
        self.id_to_class = {idx: cls for cls, idx in self.class_to_id.items()}
        
    def _create_splits(self):
        ensure_dir(self.config.splits_dir)
        
        set_seed(self.config.seed)
        
        all_images = list(set(stat['image_id'] for stat in self.validator.stats))
        random.shuffle(all_images)
        
        n_train = int(len(all_images) * self.config.train_split)
        n_val = int(len(all_images) * self.config.val_split)
        
        train_imgs = all_images[:n_train]
        val_imgs = all_images[n_train:n_train + n_val]
        test_imgs = all_images[n_train + n_val:]
        
        self.splits = {
            'train': train_imgs,
            'val': val_imgs,
            'test': test_imgs
        }
        
        for split_name, img_list in self.splits.items():
            split_path = Path(self.config.splits_dir) / f"{split_name}.txt"
            with open(split_path, 'w') as f:
                for img_id in img_list:
                    f.write(f"{img_id}\n")
        
        print(f"\nSplits created: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")
    
    def _convert_to_yolo(self):
        print("\nConverting to YOLO format...")
        ensure_dir(self.config.dataset_yolo)
        
        for split_name in ['train', 'val', 'test']:
            ensure_dir(f"{self.config.dataset_yolo}/images/{split_name}")
            ensure_dir(f"{self.config.dataset_yolo}/labels/{split_name}")
        
        for stat in tqdm(self.validator.stats, desc="Converting to YOLO"):
            try:
                self._convert_image_to_yolo(stat)
            except Exception as e:
                if self.config.verbose:
                    print(f"\nError converting {stat['image_id']}: {e}")
                    traceback.print_exc()
        
        self._create_yolo_yaml()
    
    def _convert_image_to_yolo(self, stat: Dict):
        img_id = stat['image_id']
        split = self._get_split_for_image(img_id)
        
        if split is None:
            return
        
        src_img = stat['image_path']
        dst_img = f"{self.config.dataset_yolo}/images/{split}/{img_id}{Path(src_img).suffix}"
        shutil.copy2(src_img, dst_img)
        
        mask = cv2.imread(stat['mask_path'], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return
        
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        h, w = mask_binary.shape
        contours, _ = find_contours_safe(mask_binary)
        
        label_path = f"{self.config.dataset_yolo}/labels/{split}/{img_id}.txt"
        
        with open(label_path, 'w') as f:
            for contour in contours:
                if len(contour) < 3:
                    continue
                
                epsilon = self.config.poly_epsilon
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) < 3:
                    continue
                
                class_id = self.class_to_id[stat['class']]
                
                points = approx.reshape(-1, 2)
                norm_points = points.astype(float)
                norm_points[:, 0] /= w
                norm_points[:, 1] /= h
                
                line = f"{class_id}"
                for point in norm_points:
                    line += f" {point[0]:.6f} {point[1]:.6f}"
                f.write(line + "\n")
    
    def _get_split_for_image(self, img_id: str) -> Optional[str]:
        for split_name, img_list in self.splits.items():
            if img_id in img_list:
                return split_name
        return None
    
    def _create_yolo_yaml(self):
        yaml_path = Path(self.config.dataset_yolo) / 'data.yaml'
        
        data = {
            'path': str(Path(self.config.dataset_yolo).absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.class_to_id),
            'names': self.id_to_class
        }
        
        if HAS_YAML:
            with open(yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            with open(yaml_path, 'w') as f:
                f.write(f"path: {data['path']}\n")
                f.write(f"train: {data['train']}\n")
                f.write(f"val: {data['val']}\n")
                f.write(f"test: {data['test']}\n")
                f.write(f"nc: {data['nc']}\n")
                f.write("names:\n")
                for idx, name in data['names'].items():
                    f.write(f"  {idx}: {name}\n")
        
        print(f"YOLO data.yaml saved to: {yaml_path}")
    
    def _convert_to_coco(self):
        print("\nConverting to COCO format...")
        ensure_dir(self.config.dataset_coco)
        ensure_dir(f"{self.config.dataset_coco}/annotations")
        ensure_dir(f"{self.config.dataset_coco}/images")
        
        for split_name in ['train', 'val', 'test']:
            self._create_coco_split(split_name)
    
    def _create_coco_split(self, split_name: str):
        images = []
        annotations = []
        annotation_id = 1
        
        split_imgs = self.splits[split_name]
        img_id_to_coco_id = {}
        
        for coco_img_id, img_id in enumerate(split_imgs, 1):
            img_id_to_coco_id[img_id] = coco_img_id
            
            stats_for_img = [s for s in self.validator.stats if s['image_id'] == img_id]
            if not stats_for_img:
                continue
            
            stat = stats_for_img[0]
            
            src_img = stat['image_path']
            ext = Path(src_img).suffix
            dst_img = f"{self.config.dataset_coco}/images/{img_id}{ext}"
            shutil.copy2(src_img, dst_img)
            
            images.append({
                'id': coco_img_id,
                'file_name': f"{img_id}{ext}",
                'width': stat['image_width'],
                'height': stat['image_height']
            })
        
        for stat in self.validator.stats:
            img_id = stat['image_id']
            if img_id not in split_imgs:
                continue
            
            coco_img_id = img_id_to_coco_id[img_id]
            
            mask = cv2.imread(stat['mask_path'], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            
            _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            h, w = mask_binary.shape
            contours, _ = find_contours_safe(mask_binary)
            
            for contour in contours:
                if len(contour) < 3:
                    continue
                
                segmentation = contour.reshape(-1).tolist()
                
                x, y, bw, bh = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                annotations.append({
                    'id': annotation_id,
                    'image_id': coco_img_id,
                    'category_id': self.class_to_id[stat['class']],
                    'segmentation': [segmentation],
                    'area': float(area),
                    'bbox': [x, y, bw, bh],
                    'iscrowd': 0
                })
                
                annotation_id += 1
        
        categories = [
            {'id': idx, 'name': name, 'supercategory': 'drawing_element'}
            for name, idx in self.class_to_id.items()
        ]
        
        coco_data = {
            'images': images,
            'annotations': annotations,
            'categories': categories
        }
        
        output_path = f"{self.config.dataset_coco}/annotations/instances_{split_name}.json"
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"COCO {split_name} annotations saved to: {output_path}")


class YOLOTrainer:
    def __init__(self, config: Config):
        self.config = config
        
    def train(self):
        if not HAS_ULTRALYTICS:
            print("\nUltralytics not installed. Cannot train YOLOv8.")
            print("Install with: pip install ultralytics")
            return
        
        if not HAS_TORCH:
            print("\nPyTorch not installed. Cannot train YOLOv8.")
            print("Install with: pip install torch torchvision")
            return
        
        print("\nTraining YOLOv8 segmentation model...")
        ensure_dir(self.config.models_dir)
        
        data_yaml = Path(self.config.dataset_yolo) / 'data.yaml'
        if not data_yaml.exists():
            print(f"Error: {data_yaml} not found. Run conversion first.")
            return
        
        model = YOLO('yolov8s-seg.pt')
    
        results = model.train(
            data=str(data_yaml),
            epochs=self.config.epochs,
            imgsz=self.config.imgsz,
            batch=self.config.batch,
            project=self.config.models_dir,
            name='yolov8_ignore',
            exist_ok=True,
            workers=self.config.num_workers
        )
        
        best_weights = Path(self.config.models_dir) / 'yolov8_ignore' / 'weights' / 'best.pt'
        if best_weights.exists():
            shutil.copy2(best_weights, Path(self.config.models_dir) / 'yolov8_ignore.pt')
            print(f"Best weights saved to: {Path(self.config.models_dir) / 'yolov8_ignore.pt'}")


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        mask = torch.from_numpy(mask).long()
        
        return image, mask


class SegTrainer:
    def __init__(self, config: Config, validator: DatasetValidator, converter: DatasetConverter):
        self.config = config
        self.validator = validator
        self.converter = converter
        
    def train(self):
        if not HAS_TORCH:
            print("\nPyTorch not installed. Cannot train segmentation model.")
            print("Install with: pip install torch torchvision")
            return
        
        if not HAS_SMP:
            print("\nSegmentation-models-pytorch not installed. Using basic model.")
            print("For better results, install with: pip install segmentation-models-pytorch")
        
        print("\nTraining segmentation model...")
        ensure_dir(self.config.models_dir)
        
        self._prepare_data()
        self._train_model()
    
    def _prepare_data(self):
        train_imgs = self.converter.splits['train']
        val_imgs = self.converter.splits['val']
        
        self.train_image_paths = []
        self.train_mask_paths = []
        self.val_image_paths = []
        self.val_mask_paths = []
        
        for stat in self.validator.stats:
            if stat['image_id'] in train_imgs:
                self.train_image_paths.append(stat['image_path'])
                self.train_mask_paths.append(stat['mask_path'])
            elif stat['image_id'] in val_imgs:
                self.val_image_paths.append(stat['image_path'])
                self.val_mask_paths.append(stat['mask_path'])
    
    def _train_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((512, 512)),
            T.ToTensor(),
        ])
        
        train_dataset = SegmentationDataset(self.train_image_paths, self.train_mask_paths, transform)
        val_dataset = SegmentationDataset(self.val_image_paths, self.val_mask_paths, transform)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch, shuffle=False, num_workers=2)
        
        if HAS_SMP:
            model = smp.DeepLabV3Plus(
                encoder_name="resnet50",
                encoder_weights="imagenet",
                classes=2,
                activation=None
            )
        else:
            from torchvision.models.segmentation import deeplabv3_resnet50
            model = deeplabv3_resnet50(pretrained=True)
            model.classifier[-1] = nn.Conv2d(256, 2, kernel_size=1)
        
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.lr)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            model.train()
            train_loss = 0
            
            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}"):
                images = images.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                
                if HAS_SMP:
                    outputs = model(images)
                else:
                    outputs = model(images)['out']
                
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    if HAS_SMP:
                        outputs = model(images)
                    else:
                        outputs = model(images)['out']
                    
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), Path(self.config.models_dir) / 'seg_ignore.pth')
        
        print(f"Training complete. Best model saved to: {Path(self.config.models_dir) / 'seg_ignore.pth'}")


class PreFilter:
    def __init__(self, config: Config, validator: DatasetValidator):
        self.config = config
        self.validator = validator
        self.yolo_model = None
        self.seg_model = None
        self.preserve_classes = [c.strip() for c in config.preserve_thin_classes.split(',')]

    def load_models(self):
        if self.config.weights_yolo and HAS_ULTRALYTICS:
            try:
                self.yolo_model = YOLO(self.config.weights_yolo)
                print(f"Loaded YOLOv8 model from: {self.config.weights_yolo}")
            except Exception as e:
                print(f"Failed to load YOLO model: {e}")
        
        if self.config.weights_seg and HAS_TORCH:
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                if HAS_SMP:
                    self.seg_model = smp.DeepLabV3Plus(
                        encoder_name="resnet50",
                        encoder_weights=None,
                        classes=2,
                        activation=None
                    )
                else:
                    from torchvision.models.segmentation import deeplabv3_resnet50
                    self.seg_model = deeplabv3_resnet50(pretrained=False)
                    self.seg_model.classifier[-1] = nn.Conv2d(256, 2, kernel_size=1)
                
                self.seg_model.load_state_dict(torch.load(self.config.weights_seg, map_location=device))
                self.seg_model = self.seg_model.to(device)
                self.seg_model.eval()
                print(f"Loaded segmentation model from: {self.config.weights_seg}")
            except Exception as e:
                print(f"Failed to load segmentation model: {e}")

    def run_inference(self):
        self.load_models()
        
        print("\nRunning pre-filter inference...")
        output_base = Path(self.config.output_dir) / 'dataset_cleaned'
        
        if not self.config.dry_run:
            ensure_dir(str(output_base))
        
        ensure_dir(Path(self.config.output_dir) / 'masked_overlays')
        ensure_dir(Path(self.config.output_dir) / 'final_masks')
        
        if self.config.mask_mode == 'crop_regions':
            ensure_dir(Path(self.config.output_dir) / 'crops')
        
        processed_count = 0
        removed_count = 0
        
        image_dict = defaultdict(list)
        for stat in self.validator.stats:
            image_dict[stat['image_id']].append(stat)
        
        for img_id, stats in tqdm(image_dict.items(), desc="Processing images"):
            try:
                result = self._process_image(img_id, stats, output_base)
                processed_count += 1
                if result:
                    removed_count += 1
            except Exception as e:
                if self.config.verbose:
                    print(f"\nError processing {img_id}: {e}")
                    traceback.print_exc()
        
        print(f"\nProcessing complete:")
        print(f"  Images processed: {processed_count}")
        print(f"  Images with removals: {removed_count}")

    def _process_image(self, img_id: str, stats: List[Dict], output_base: Path) -> bool:
        stat = stats[0]
        image = cv2.imread(stat['image_path'])
        if image is None:
            return False
            
        h, w = image.shape[:2]
        
        ignore_classes = []
        if self.config.ignore_classes:
            ignore_classes = [c.strip() for c in self.config.ignore_classes.split(',')]
        else:
            ignore_classes = [s['class'] for s in stats if s['category'] == 'Non_essential']
        
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        if self.yolo_model:
            yolo_mask = self._run_yolo_inference(image, ignore_classes)
            if yolo_mask is not None:
                combined_mask = cv2.bitwise_or(combined_mask, yolo_mask)
        
        if self.seg_model and self.config.use_seg_refinement:
            seg_mask = self._run_seg_inference(image)
            if seg_mask is not None:
                combined_mask = cv2.bitwise_or(combined_mask, seg_mask)
        
        combined_mask = self._morphological_cleanup(combined_mask)
        
        preserve_mask = self._get_preserve_mask(stats)
        if preserve_mask is not None:
            combined_mask = cv2.bitwise_and(combined_mask, cv2.bitwise_not(preserve_mask))
        
        mask_area = np.sum(combined_mask > 0)
        mask_fraction = mask_area / (w * h)
        
        should_apply = self.config.force_remove or (mask_fraction >= self.config.remove_threshold)
        
        self._save_overlay(image, combined_mask, img_id)
        
        mask_path = Path(self.config.output_dir) / 'final_masks' / f"{img_id}_mask.png"
        cv2.imwrite(str(mask_path), combined_mask)
        
        if should_apply and not self.config.dry_run:
            if self.config.mask_mode == 'blackout':
                cleaned = image.copy()
                cleaned[combined_mask > 0] = [255, 255, 255]
                
                relative_path = Path(stat['image_path']).relative_to(self.config.dataset_root)
                output_path = output_base / relative_path
                ensure_dir(str(output_path.parent))
                cv2.imwrite(str(output_path), cleaned)
                
            elif self.config.mask_mode == 'crop_regions':
                contours, _ = find_contours_safe(combined_mask)
                crops_dir = Path(self.config.output_dir) / 'crops'
                
                for idx, contour in enumerate(contours):
                    x, y, bw, bh = cv2.boundingRect(contour)
                    crop = image[y:y+bh, x:x+bw]
                    crop_path = crops_dir / f"{img_id}_region_{idx}.png"
                    cv2.imwrite(str(crop_path), crop)
        
        return should_apply

    def _run_yolo_inference(self, image, ignore_classes):
        try:
            results = self.yolo_model(image, verbose=False)
            h, w = image.shape[:2]
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            
            for result in results:
                if result.masks is None:
                    continue
                
                masks = result.masks.data.cpu().numpy()
                
                for mask in masks:
                    mask_resized = cv2.resize(mask, (w, h))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                    combined_mask = cv2.bitwise_or(combined_mask, mask_binary)
            
            return combined_mask
        except Exception as e:
            if self.config.verbose:
                print(f"YOLO inference error: {e}")
            return None

    def _run_seg_inference(self, image):
        try:
            device = next(self.seg_model.parameters()).device
            
            h, w = image.shape[:2]
            input_image = cv2.resize(image, (512, 512))
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            input_tensor = torch.from_numpy(input_image).permute(2, 0, 1).float() / 255.0
            input_tensor = input_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                if HAS_SMP:
                    output = self.seg_model(input_tensor)
                else:
                    output = self.seg_model(input_tensor)['out']
                
                pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
                pred_resized = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            
            mask = (pred_resized > 0).astype(np.uint8) * 255
            
            return mask
        except Exception as e:
            if self.config.verbose:
                print(f"Segmentation inference error: {e}")
            return None

    def _morphological_cleanup(self, mask):
        contours, _ = find_contours_safe(mask)
        
        cleaned = np.zeros_like(mask)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.config.min_area_px:
                cv2.drawContours(cleaned, [contour], -1, 255, -1)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        return cleaned

    def _get_preserve_mask(self, stats):
        stat = stats[0]
        image = cv2.imread(stat['image_path'])
        if image is None:
            return None
            
        h, w = image.shape[:2]
        preserve_mask = np.zeros((h, w), dtype=np.uint8)
        
        for s in stats:
            if s['class'] in self.preserve_classes:
                mask = cv2.imread(s['mask_path'], cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    preserve_mask = cv2.bitwise_or(preserve_mask, mask_binary)
        
        return preserve_mask if np.any(preserve_mask) else None

    def _save_overlay(self, image, mask, img_id):
        overlay = image.copy()
        overlay[mask > 0] = [0, 0, 255]
        
        blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        
        overlay_path = Path(self.config.output_dir) / 'masked_overlays' / f"{img_id}_overlay.png"
        cv2.imwrite(str(overlay_path), blended)


class PreviewGenerator:
    def __init__(self, config: Config, validator: DatasetValidator):
        self.config = config
        self.validator = validator

    def generate_previews(self):
        print(f"\nGenerating previews for first {self.config.preview_n} images...")
        ensure_dir(Path(self.config.output_dir) / 'preview')
        
        processed = 0
        for stat in self.validator.stats:
            if processed >= self.config.preview_n:
                break
            
            try:
                self._generate_preview(stat)
                processed += 1
            except Exception as e:
                if self.config.verbose:
                    print(f"\nError generating preview for {stat['image_id']}: {e}")

    def _generate_preview(self, stat: Dict):
        image = cv2.imread(stat['image_path'])
        mask = cv2.imread(stat['mask_path'], cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            return
        
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        overlay = image.copy()
        overlay[mask_binary > 0] = [0, 255, 0]
        
        blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        
        preview_path = Path(self.config.output_dir) / 'preview' / f"{stat['image_id']}_preview.png"
        cv2.imwrite(str(preview_path), blended)


def main():
    parser = argparse.ArgumentParser(description="Engineering Drawing Dataset Pre-filter Pipeline")

    parser.add_argument('--mode', type=str, default='help',
                      choices=['stats', 'convert', 'train', 'infer', 'full_pipeline', 'preview', 'help'],
                      help='Operation mode')
    parser.add_argument('--dataset_root', type=str, default='./dataset')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--dataset_normalized', type=str, default='./dataset_normalized')
    parser.add_argument('--dataset_yolo', type=str, default='./dataset_yolo')
    parser.add_argument('--dataset_coco', type=str, default='./dataset_coco')
    parser.add_argument('--splits_dir', type=str, default='./splits')
    parser.add_argument('--models_dir', type=str, default='./models')
    parser.add_argument('--weights_yolo', type=str, default=None)
    parser.add_argument('--weights_seg', type=str, default=None)
    parser.add_argument('--train_yolov8', action='store_true')
    parser.add_argument('--train_seg', action='store_true')
    parser.add_argument('--convert_format', type=str, default='yolo', choices=['yolo', 'coco'])
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--remove_threshold', type=float, default=0.02)
    parser.add_argument('--min_area_px', type=int, default=50)
    parser.add_argument('--preserve_thin_classes', type=str,
                        default='Center_line,hidden_lines,Phantom_lines,Feature_Visible,Section hatching (cross-hatch)')
    parser.add_argument('--mask_mode', type=str, default='blackout',
                        choices=['blackout', 'crop_regions', 'save_mask'])
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--preview_n', type=int, default=10)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--ignore_classes', type=str, default=None)
    parser.add_argument('--force_remove', action='store_true')
    parser.add_argument('--use_seg_refinement', action='store_true')
    parser.add_argument('--poly_epsilon', type=float, default=2.0)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=6)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    config = Config(**vars(args))

    if config.config:
        yaml_config = load_config_from_yaml(config.config)
        for key, value in yaml_config.items():
            setattr(config, key, value)

    if config.mode == 'help':
        parser.print_help()
        # ... (keep existing help print) ...
        return

    print("\n" + "="*80)
    print("Engineering Drawing Dataset Pre-filter Pipeline")
    print("="*80)
    
    set_seed(config.seed)
    validator = DatasetValidator(config)

    # --- OPTIMIZED LOGIC START ---
    
    # Helper: Load stats if available, otherwise compute them
    def ensure_stats(force_recompute=False):
        if not force_recompute and validator.load_stats_from_csv():
            return
        validator.validate_and_compute_stats()

    if config.mode == 'stats':
        # Force recompute to ensure freshness
        ensure_stats(force_recompute=True)
        DatasetNormalizer(config, validator).normalize_masks()

    elif config.mode == 'convert':
        # Load existing stats if possible
        ensure_stats(force_recompute=False)
        # We skip normalization here if you assume it's done, 
        # but it's safer to run it (it's fast compared to validation).
        DatasetNormalizer(config, validator).normalize_masks()
        DatasetConverter(config, validator).convert()

    elif config.mode == 'train':
        ensure_stats(force_recompute=False)
        # Check if YOLO data exists to skip conversion
        yolo_data_exists = (Path(config.dataset_yolo) / 'data.yaml').exists()
        
        if not yolo_data_exists:
            DatasetNormalizer(config, validator).normalize_masks()
            DatasetConverter(config, validator).convert()
        else:
            print("YOLO dataset found. Skipping conversion step.")
        
        if config.train_yolov8:
            YOLOTrainer(config).train()
        
        if config.train_seg:
            # SegTrainer needs the converter to access split lists
            converter = DatasetConverter(config, validator)
            converter._build_class_mapping() # Lightweight setup
            converter._create_splits()       # Lightweight setup
            SegTrainer(config, validator, converter).train()

    elif config.mode == 'infer':
        ensure_stats(force_recompute=False)
        PreFilter(config, validator).run_inference()

    elif config.mode == 'full_pipeline':
        ensure_stats(force_recompute=True) # Full pipeline usually implies fresh start
        DatasetNormalizer(config, validator).normalize_masks()
        converter = DatasetConverter(config, validator)
        converter.convert()
        
        if config.train_yolov8:
            YOLOTrainer(config).train()
        
        if config.train_seg:
            SegTrainer(config, validator, converter).train()
            
        PreFilter(config, validator).run_inference()

    elif config.mode == 'preview':
        ensure_stats(force_recompute=False)
        PreviewGenerator(config, validator).generate_previews()

    print("\n" + "="*80)
    print("Pipeline complete!")
    print("="*80)

if __name__ == '__main__':
    main()