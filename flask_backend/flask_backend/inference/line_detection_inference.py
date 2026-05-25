"""
Line Detection Inference Module
-------------------------------
Inference logic for tiled sliding-window segmentation
of engineering drawing line types.

Matches the exact pipeline from inference_step3.py.

Default deliverables (matching inference_step3.py without --debug):
- masks/{basename}_mask.png
- binary_masks/{basename}_class{N}.png
- processing_stats.json
"""

import os
import sys
import time
import importlib.util
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from PIL import Image

# Add parent directories for imports
APPROACH2_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(APPROACH2_DIR))

# Import from the backend modules
from postprocess import (
    process_single_image,
    get_default_class_thresholds, get_default_morph_params,
    convert_thresholds_to_indices, convert_morph_to_indices
)
from crf_utils import is_crf_available

# Add flask_backend to path for Config import
FLASK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FLASK_DIR))
from config import Config
from utils_io import (
    load_json, save_json, get_class_colors,
    generate_tiles, extract_tile, paste_tile,
    create_gaussian_weight_window
)


def load_model_module():
    """Load model.py from Approach2 directory."""
    model_path = APPROACH2_DIR / "model.py"
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    return model_module


class LineDetectionInference:
    """
    Line Detection Inference class that EXACTLY matches inference_step3.py.

    Key features ported from inference_step3.py:
    - Temperature scaling (default 0.5)
    - Adaptive preprocessing (CLAHE for low-contrast)
    - Full TTA (4 augmentations)
    - Complete post-processing pipeline (CRF/bilateral, edge boost, force-argmax)
    - Per-class thresholds and morphological operations
    - White padding for tiles
    """

    def __init__(
        self,
        model_path: str,
        legend_path: str = None,
        num_classes: int = 11,
        backbone: str = "resnet50",
        device: str = "cuda",
        tile_size: int = 1024,
        overlap: int = 256,
        temperature: float = 0.5,  # CRITICAL: temperature scaling
        class_names: List[str] = None,
        class_colors: Dict[str, List[int]] = None
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.backbone = backbone
        self.tile_size = tile_size
        self.overlap = overlap
        self.temperature = temperature  # CRITICAL: added temperature

        # Default class configuration
        self.class_names = class_names or [
            "background", "center_line", "dimension_line", "extension_line",
            "feature_visible", "leader_line", "phantom_line", "section_hatching",
            "break_line", "cutting_plane", "hidden_line"
        ]

        self.class_colors = class_colors or {
            "background": [0, 0, 0],
            "center_line": [255, 0, 0],
            "dimension_line": [0, 255, 0],
            "extension_line": [0, 0, 255],
            "feature_visible": [255, 255, 0],
            "leader_line": [255, 0, 255],
            "section_hatching": [255, 165, 0],
            "cutting_plane": [0, 128, 128],
        }

        # Excluded class names (phantom_line, break_line, hidden_line)
        self.excluded_class_names = Config.EXCLUDED_CLASS_NAMES
        Config._init_excluded_indices()
        self.excluded_class_indices = Config.EXCLUDED_CLASS_INDICES

        # Normalization parameters (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Load legend
        self.legend = self._load_legend(legend_path)

        # Load thresholds and morph params (from postprocess.py defaults)
        thresh_by_name = get_default_class_thresholds()
        self.class_thresholds = convert_thresholds_to_indices(thresh_by_name, self.legend)

        morph_by_name = get_default_morph_params()
        self.morph_params = convert_morph_to_indices(morph_by_name, self.legend)

        # Dynamically import create_model from model.py
        model_module = load_model_module()
        self.create_model = model_module.create_model
        self.model = self._load_model(model_path)

    def _load_legend(self, legend_path: str) -> Dict:
        """Load legend.json or create default."""
        if legend_path and os.path.exists(legend_path):
            return load_json(legend_path)

        # Create default legend
        return {
            'class_to_index': {name: idx for idx, name in enumerate(self.class_names)},
            'class_to_color': {
                name: '#{:02x}{:02x}{:02x}'.format(*color)
                for name, color in self.class_colors.items()
            }
        }

    def _load_model(self, model_path: str):
        """Load the trained model (matches inference_step3.py load_model)."""
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}")
            return None

        ext = Path(model_path).suffix.lower()

        if ext in ['.pt', '.pth']:
            # Try TorchScript first
            try:
                model = torch.jit.load(model_path, map_location=self.device)
                model.eval()
                print(f"Line detection model loaded as TorchScript from {model_path}")
                return model
            except:
                pass

            # Load as checkpoint
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

            if 'model' in checkpoint:
                model = checkpoint['model']
            else:
                model = self.create_model(num_classes=self.num_classes, backbone=self.backbone)
                state_dict_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'state_dict'
                ckpt_sd = checkpoint.get(state_dict_key, checkpoint)
                
                # Remap keys if checkpoint is a pure SMP model (missing segmentation_model prefix)
                mapped_sd = {}
                for k, v in ckpt_sd.items():
                    if k.startswith(('encoder.', 'decoder.', 'segmentation_head.')):
                        mapped_sd[f'segmentation_model.{k}'] = v
                    else:
                        mapped_sd[k] = v
                
                try:
                    model.load_state_dict(mapped_sd, strict=False)
                except Exception as e:
                    print(f"Warning during state_dict load: {e}")

            model = model.to(self.device)
            model.eval()
            print(f"Line detection model loaded from {model_path}")
            return model

        raise ValueError(f"Unsupported model format: {ext}")

    def preprocess_image_adaptive(self, image: np.ndarray, enhance_contrast: bool = True) -> np.ndarray:
        """
        Adaptive preprocessing to handle different drawing styles.
        Matches inference_step3.py preprocess_image_adaptive()
        """
        if not enhance_contrast:
            return image

        # Convert to grayscale to analyze
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Check contrast
        contrast = gray.std()

        # If low contrast, apply CLAHE enhancement
        if contrast < 50:
            print(f"  Low contrast detected ({contrast:.1f}), applying CLAHE enhancement...")
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return image

    def _run_model_inference(
        self,
        tile: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run model inference on a single tile with temperature scaling.
        Matches inference_step3.py run_model_inference()
        """
        # Preprocess
        tile_float = tile.astype(np.float32) / 255.0
        tile_norm = (tile_float - self.mean) / self.std
        tile_tensor = tile_norm.transpose(2, 0, 1)[np.newaxis, ...]

        with torch.no_grad():
            input_tensor = torch.from_numpy(tile_tensor).float().to(self.device)
            outputs = self.model(input_tensor)

            if isinstance(outputs, tuple):
                class_logits, edge_logits = outputs[0], outputs[1]
            elif isinstance(outputs, dict):
                if 'seg_logits' in outputs:
                    class_logits = outputs['seg_logits']
                elif 'class' in outputs:
                    class_logits = outputs['class']
                else:
                    raise KeyError(f"Missing class logits. Keys: {list(outputs.keys())}")

                edge_logits = outputs.get('edge_logits', outputs.get('edge', None))
            else:
                class_logits = outputs
                edge_logits = None

            # CRITICAL: Apply temperature scaling (lower = sharper predictions)
            if self.temperature != 1.0:
                class_logits = class_logits / self.temperature

            prob_class = torch.softmax(class_logits, dim=1).squeeze(0).cpu().numpy()

            if edge_logits is not None:
                prob_edge = torch.sigmoid(edge_logits).squeeze().cpu().numpy()
            else:
                prob_edge = np.zeros(prob_class.shape[1:], dtype=np.float32)

        return prob_class, prob_edge

    def _run_tta_inference(
        self,
        tile: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference with FULL Test Time Augmentation (4 augmentations).
        Matches inference_step3.py run_tta_inference()
        """
        predictions_class = []
        predictions_edge = []

        # Original
        pc, pe = self._run_model_inference(tile)
        predictions_class.append(pc)
        predictions_edge.append(pe)

        # Horizontal flip
        tile_hflip = np.fliplr(tile).copy()
        pc, pe = self._run_model_inference(tile_hflip)
        predictions_class.append(np.flip(pc, axis=2))
        predictions_edge.append(np.fliplr(pe))

        # Vertical flip
        tile_vflip = np.flipud(tile).copy()
        pc, pe = self._run_model_inference(tile_vflip)
        predictions_class.append(np.flip(pc, axis=1))
        predictions_edge.append(np.flipud(pe))

        # 90 degree rotation
        tile_rot90 = np.rot90(tile).copy()
        pc, pe = self._run_model_inference(tile_rot90)
        predictions_class.append(np.rot90(pc, k=-1, axes=(1, 2)))
        predictions_edge.append(np.rot90(pe, k=-1))

        return np.mean(predictions_class, axis=0), np.mean(predictions_edge, axis=0)

    def run_tiled_inference(
        self,
        image: np.ndarray,
        use_tta: bool = False,
        gaussian_blend: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run tiled sliding-window inference.
        Matches inference_step3.py run_tiled_inference()
        """
        if self.model is None:
            return None, None

        self.model.eval()
        h, w = image.shape[:2]

        # Create Gaussian weight window (matches inference_step3.py)
        if gaussian_blend:
            weight_window = create_gaussian_weight_window(self.tile_size)
        else:
            weight_window = np.ones((self.tile_size, self.tile_size), dtype=np.float32)

        prob_accum = np.zeros((self.num_classes, h, w), dtype=np.float32)
        edge_accum = np.zeros((h, w), dtype=np.float32)
        weight_accum = np.zeros((h, w), dtype=np.float32)

        # Generate tile coordinates
        tile_coords = list(generate_tiles((h, w), self.tile_size, self.overlap))

        for coords in tile_coords:
            # Extract tile with WHITE padding (255) - CRITICAL difference from Flask
            tile = extract_tile(image, coords, self.tile_size, pad_value=255)

            if use_tta:
                prob_class, prob_edge = self._run_tta_inference(tile)
            else:
                prob_class, prob_edge = self._run_model_inference(tile)

            # Paste with weighted accumulation
            paste_tile(prob_accum, weight_accum, prob_class, weight_window, coords)

            y_start, y_end, x_start, x_end = coords
            tile_h, tile_w = y_end - y_start, x_end - x_start
            edge_cropped = prob_edge[:tile_h, :tile_w]
            weight_cropped = weight_window[:tile_h, :tile_w]
            edge_accum[y_start:y_end, x_start:x_end] += edge_cropped * weight_cropped

        # Normalize by weights
        weight_accum = np.maximum(weight_accum, 1e-8)
        for c in range(self.num_classes):
            prob_accum[c] /= weight_accum
        edge_accum /= weight_accum

        return prob_accum, edge_accum

    def process_image(
        self,
        image_path: str,
        output_dir: str,
        use_tta: bool = False,
        enhance_contrast: bool = True,
        use_crf: bool = False,
        use_bilateral: bool = True,
        alpha: float = 2.0,
        default_threshold: float = 0.08,
        force_argmax: bool = True,
        min_confidence: float = 0.02
    ) -> Dict[str, Any]:
        """
        Process a single image with the FULL pipeline from inference_step3.py.

        Produces only the default deliverables (matching inference_step3.py
        without --debug flag):
          - masks/{basename}_mask.png
          - binary_masks/{basename}_class{N}.png
        """
        start_time = time.time()

        # Load image
        image = np.array(Image.open(image_path).convert('RGB'))
        filename = os.path.basename(image_path)
        basename = os.path.splitext(filename)[0]

        h, w = image.shape[:2]
        print(f"  Image size: {w}x{h} ({w*h:,} pixels)")

        # Adaptive contrast enhancement (from inference_step3.py)
        if enhance_contrast:
            image = self.preprocess_image_adaptive(image, enhance_contrast=True)

        # Create output directories (only masks and binary_masks)
        masks_dir = os.path.join(output_dir, 'masks')
        binary_masks_dir = os.path.join(output_dir, 'binary_masks')

        for d in [masks_dir, binary_masks_dir]:
            os.makedirs(d, exist_ok=True)

        # Run inference
        inference_start = time.time()
        prob_class, prob_edge = self.run_tiled_inference(image, use_tta=use_tta)
        inference_time = time.time() - inference_start

        if prob_class is None:
            return {
                'filename': filename,
                'success': False,
                'error': 'Model not loaded',
                'processing_time': time.time() - start_time
            }

        # CRITICAL: Zero out excluded class probability channels
        for c in self.excluded_class_indices:
            if c < prob_class.shape[0]:
                prob_class[c, :, :] = 0.0

        # Print probability statistics (from inference_step3.py)
        self._print_probability_stats(prob_class)

        # Full post-processing pipeline from inference_step3.py
        post_start = time.time()

        crf_params = {'tile_size': 1024}

        argmax_mask, binary_masks, skeleton, stats = process_single_image(
            image=image,
            prob_class=prob_class,
            prob_edge=prob_edge,
            legend=self.legend,
            alpha=alpha,
            normalize=False,
            use_crf=use_crf,
            use_bilateral=use_bilateral,
            crf_params=crf_params,
            class_thresholds=self.class_thresholds,
            morph_params=self.morph_params,
            default_threshold=default_threshold,
            force_argmax=force_argmax,
            min_confidence=min_confidence
        )
        post_time = time.time() - post_start

        # Report detected classes (matches inference_step3.py console output)
        unique_classes = np.unique(argmax_mask)
        print(f"\n  Detected classes in output: {[c for c in unique_classes.tolist() if c not in self.excluded_class_indices]}")
        for c in unique_classes:
            if c > 0 and c not in self.excluded_class_indices:
                count = (argmax_mask == c).sum()
                print(f"    Class {c}: {count:,} pixels")

        # Save outputs (matching inference_step3.py defaults)
        # 1. Segmentation mask
        mask_path = os.path.join(masks_dir, f'{basename}_mask.png')
        Image.fromarray(argmax_mask).save(mask_path)

        # 2. Binary masks per class (skip background and excluded classes)
        for class_idx, mask in binary_masks.items():
            if class_idx == 0 or class_idx in self.excluded_class_indices:
                continue
            mask_file = os.path.join(binary_masks_dir, f'{basename}_class{class_idx}.png')
            Image.fromarray(mask).save(mask_file)

        processing_time = time.time() - start_time

        # Add timing info to stats (matches inference_step3.py)
        stats['basename'] = basename
        stats['inference_time'] = inference_time
        stats['postprocess_time'] = post_time
        stats['total_time'] = inference_time + post_time
        stats['image_size'] = [h, w]

        print(f"  Timing: inference={inference_time:.2f}s, postprocess={post_time:.2f}s")

        return {
            'filename': filename,
            'success': True,
            'paths': {
                'mask': mask_path,
                'binary_masks': binary_masks_dir
            },
            'stats': stats,
            'timings': {
                'inference': round(inference_time, 2),
                'postprocess': round(post_time, 2),
                'total': round(processing_time, 2)
            },
            # Internal arrays for UI display generation (not saved to disk)
            '_image': image,
            '_argmax_mask': argmax_mask,
            '_prob_class': prob_class,
            '_skeleton': skeleton
        }

    def _print_probability_stats(self, prob_class: np.ndarray):
        """Print detailed probability statistics (from inference_step3.py), skipping excluded classes."""
        n_classes = prob_class.shape[0]

        print("\n  Probability Statistics:")
        print("  " + "-" * 60)

        total_nonbg = 0
        for c in range(n_classes):
            if c in self.excluded_class_indices:
                continue

            class_name = self.class_names[c] if c < len(self.class_names) else f'class_{c}'

            max_p = prob_class[c].max()
            mean_p = prob_class[c].mean()
            above_01 = (prob_class[c] > 0.1).sum()
            above_03 = (prob_class[c] > 0.3).sum()

            if c > 0:
                total_nonbg += above_01

            print(f"  {c:2d} {class_name:20s}: max={max_p:.4f}, mean={mean_p:.6f}, "
                  f">0.1:{above_01:8,}, >0.3:{above_03:8,}")

        print("  " + "-" * 60)
        print(f"  Total non-background pixels (>0.1): {total_nonbg:,}")

        if total_nonbg == 0:
            print("\n  WARNING: Model detected NO lines! Possible issues:")
            print("    - Model not trained for this drawing type")
            print("    - Drawing needs contrast enhancement")
            print("    - Try using TTA (test-time augmentation)")
