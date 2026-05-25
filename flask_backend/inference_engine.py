"""
Inference Engine aligned with inference.py structure.
Supports tiled inference, temperature scaling, TTA, and visualization generation.
"""

import os
import sys
import json
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import importlib.util

import torch
import torch.nn.functional as F

FLASK_DIR = Path(__file__).parent
APPROACH2_DIR = FLASK_DIR.parent
sys.path.insert(0, str(APPROACH2_DIR))
sys.path.insert(0, str(FLASK_DIR))

from config import Config

# Lazy import SMP to avoid CUDA initialization crash on app startup
smp = None


def get_smp():
    """Lazy-load segmentation_models_pytorch on first use."""
    global smp
    if smp is None:
        try:
            import segmentation_models_pytorch
            smp = segmentation_models_pytorch
        except ImportError:
            print("ERROR: Install segmentation-models-pytorch: pip install segmentation-models-pytorch")
            raise
    return smp


class InferenceEngine:
    """Line detection inference engine matching inference.py structure."""

    def __init__(
        self,
        line_detection_model_path: str = None,
        legend_path: str = None,
        num_classes: int = None,
        backbone: str = None,
        device: str = None,
        tile_size: int = None,
        overlap: int = None,
        temperature: float = None,
    ):
        # Use Config defaults if not provided
        num_classes = num_classes or Config.NUM_CLASSES
        backbone = backbone or Config.BACKBONE
        device = device or Config.DEVICE
        tile_size = tile_size or Config.TILE_SIZE
        overlap = overlap or Config.OVERLAP
        temperature = temperature or Config.TEMPERATURE

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.backbone = backbone
        self.tile_size = tile_size
        self.overlap = overlap
        self.temperature = temperature
        self._current_checkpoint = None
        self.model = None

        if line_detection_model_path is None:
            line_detection_model_path = Config.DEFAULT_CHECKPOINT

        self.legend_path = legend_path or str(APPROACH2_DIR / "legend.json")

        # Defer model loading - will be done on first use or explicit call
        self._pending_model_path = line_detection_model_path

    def ensure_model_loaded(self):
        """Lazy-load model on first use."""
        if self.model is None and self._pending_model_path:
            self._load_model(self._pending_model_path)
            self._pending_model_path = None

    def _load_model(self, model_path: str):
        """Load model checkpoint (matches inference.py logic)."""
        if not model_path or not os.path.exists(model_path):
            print(f"Warning: model not found at {model_path}")
            self.model = None
            return

        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=self.device)
        except RuntimeError as e:
            # CUDA out of memory or initialization error - fallback to CPU
            if 'cuda' in str(e).lower() or 'out of memory' in str(e).lower():
                print(f"CUDA error detected: {e}")
                print("Falling back to CPU...")
                self.device = torch.device("cpu")
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            else:
                raise

        # Determine state dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            epoch = checkpoint.get("epoch", "?")
            miou = checkpoint.get("best_miou", "?")
        else:
            state_dict = checkpoint
            epoch = "?"
            miou = "?"

        # Auto-detect num_classes
        detected_classes = None
        for key in state_dict:
            if "segmentation_head" in key and "weight" in key:
                detected_classes = state_dict[key].shape[0]
                break

        if detected_classes is not None:
            self.num_classes = detected_classes

        # Check if wrapped model or raw SMP
        has_base_model = any(k.startswith("base_model.") for k in state_dict)
        has_edge_head = any(k.startswith("edge_head.") for k in state_dict)

        try:
            if has_base_model or has_edge_head:
                # Load DeepLabV3PlusEdge from model.py
                model_spec = importlib.util.spec_from_file_location(
                    "model_module", str(APPROACH2_DIR / "model.py")
                )
                model_module = importlib.util.module_from_spec(model_spec)
                model_spec.loader.exec_module(model_module)
                self.model = model_module.DeepLabV3PlusEdge(
                    num_classes=self.num_classes,
                    backbone=self.backbone,
                    use_edge_head=has_edge_head,
                )
                self.model.load_state_dict(state_dict, strict=False)
            else:
                # Load raw SMP DeepLabV3Plus
                smp_module = get_smp()
                self.model = smp_module.DeepLabV3Plus(
                    encoder_name=self.backbone,
                    encoder_weights=None,
                    in_channels=3,
                    classes=self.num_classes,
                )
                self.model.load_state_dict(state_dict, strict=False)

            self.model = self.model.to(self.device)
            self.model.eval()
            self._current_checkpoint = model_path
            print(f"Model loaded: {model_path} ({self.num_classes} classes, epoch={epoch}, miou={miou})")

        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    def switch_model(self, checkpoint_path: str) -> bool:
        """Hot-swap model checkpoint."""
        if checkpoint_path == self._current_checkpoint:
            return True
        if not os.path.exists(checkpoint_path):
            return False
        self._load_model(checkpoint_path)
        return self.model is not None

    # ── Inference helpers (from inference.py) ────────────────────────

    def _get_gaussian_window(self, size: int, sigma: float = 0.5) -> np.ndarray:
        """Create Gaussian window for blending."""
        x = np.linspace(-1, 1, size)
        g = np.exp(-0.5 * (x / sigma) ** 2)
        return np.outer(g, g).astype(np.float32)

    def _run_model_inference(self, tile: np.ndarray) -> np.ndarray:
        """Run model on single tile with temperature scaling."""
        # Normalize (ImageNet)
        tile_float = tile.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tile_norm = (tile_float - mean) / std
        tile_tensor = tile_norm.transpose(2, 0, 1)[np.newaxis, ...]

        with torch.no_grad():
            input_tensor = torch.from_numpy(tile_tensor).float().to(self.device)
            outputs = self.model(input_tensor)

            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, dict):
                logits = outputs.get('seg_logits', outputs.get('class', outputs))
            else:
                logits = outputs

            # Temperature scaling (lower = sharper)
            if self.temperature != 1.0:
                logits = logits / self.temperature

            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        return probs

    def _run_tta_inference(self, tile: np.ndarray) -> np.ndarray:
        """Run inference with 4 TTA augmentations."""
        predictions = []

        # Original
        predictions.append(self._run_model_inference(tile))

        # Horizontal flip
        tile_hflip = np.fliplr(tile).copy()
        pred_hflip = self._run_model_inference(tile_hflip)
        predictions.append(np.flip(pred_hflip, axis=2))

        # Vertical flip
        tile_vflip = np.flipud(tile).copy()
        pred_vflip = self._run_model_inference(tile_vflip)
        predictions.append(np.flip(pred_vflip, axis=1))

        # 90 degree rotation
        tile_rot = np.rot90(tile).copy()
        pred_rot = self._run_model_inference(tile_rot)
        predictions.append(np.rot90(pred_rot, k=-1, axes=(1, 2)))

        return np.mean(predictions, axis=0)

    def run_tiled_inference(self, image: np.ndarray, use_tta: bool = False) -> np.ndarray:
        """Run tiled inference with Gaussian blending (from inference.py)."""
        self.ensure_model_loaded()

        if self.model is None:
            return None

        h, w = image.shape[:2]
        stride = self.tile_size - self.overlap
        window = self._get_gaussian_window(self.tile_size)

        prob_map = np.zeros((self.num_classes, h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y2 = min(y + self.tile_size, h)
                x2 = min(x + self.tile_size, w)
                y1 = max(0, y2 - self.tile_size)
                x1 = max(0, x2 - self.tile_size)

                tile = image[y1:y2, x1:x2]
                th, tw = tile.shape[:2]

                # Pad with black (0)
                if th < self.tile_size or tw < self.tile_size:
                    pad_tile = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
                    pad_tile[:th, :tw] = tile
                    tile = pad_tile

                probs = self._run_tta_inference(tile) if use_tta else self._run_model_inference(tile)

                tile_h = min(self.tile_size, y2 - y1)
                tile_w = min(self.tile_size, x2 - x1)
                probs = probs[:, :tile_h, :tile_w]
                win = window[:tile_h, :tile_w]

                prob_map[:, y1:y2, x1:x2] += probs * win
                count_map[y1:y2, x1:x2] += win

        prob_map /= (count_map[np.newaxis, :, :] + 1e-6)
        return prob_map

    # ── Visualization (from inference.py - EXACT MATCH) ────────────────

    def _create_colored_overlay(self, image_rgb: np.ndarray, pred_mask: np.ndarray,
                                 alpha: float = 0.45) -> np.ndarray:
        """Create overlay with all classes colored (matches inference.py exactly)."""
        colored = np.zeros_like(image_rgb)

        for cls_id in range(1, self.num_classes):
            if cls_id in Config.EXCLUDED_CLASSES:
                continue
            color_rgb = Config.CLASS_COLORS_RGB.get(cls_id, [200, 200, 200])
            colored[pred_mask == cls_id] = color_rgb

        overlay = cv2.addWeighted(image_rgb, 1.0 - alpha, colored, alpha, 0)
        return overlay

    def _create_per_class_masks(self, image_rgb: np.ndarray, pred_mask: np.ndarray) -> Dict:
        """Create individual views for each detected class (matches inference.py exactly)."""
        masks = {}
        for cls_id in range(1, self.num_classes):
            if cls_id in Config.EXCLUDED_CLASSES:
                continue

            cls_mask = (pred_mask == cls_id).astype(np.uint8)
            pixel_count = cls_mask.sum()

            if pixel_count == 0:
                continue

            name = Config.CLASS_NAMES[cls_id]
            color_rgb = Config.CLASS_COLORS_RGB.get(cls_id, [200, 200, 200])

            # Dimmed original + bright colored class pixels (30% original, 70% color)
            view = (image_rgb * 0.3).astype(np.uint8)
            mask_3ch = np.stack([cls_mask] * 3, axis=-1)
            colored_pixels = np.array(color_rgb, dtype=np.uint8)
            view = np.where(mask_3ch > 0, colored_pixels, view)

            # Add label
            label = f"{name} (ID:{cls_id}, {pixel_count:,}px)"
            cv2.putText(view, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 255), 2, cv2.LINE_AA)

            masks[cls_id] = {"name": name, "view": view, "pixels": pixel_count}

        return masks

    def _create_confidence_heatmap(self, prob_map: np.ndarray,
                                    image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create heatmap showing prediction confidence (matches inference.py)."""
        confidence = np.max(prob_map, axis=0)

        # Normalize and colorize
        conf_uint8 = (confidence * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(conf_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Blend with original (40% original, 60% heatmap)
        blended = cv2.addWeighted(image_rgb, 0.4, heatmap_rgb, 0.6, 0)

        # Add colorbar text
        cv2.putText(blended, "Low Conf", (15, blended.shape[0] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(blended, "High Conf", (blended.shape[1] - 130, blended.shape[0] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return blended, confidence

    def _create_summary_panel(self, image_rgb: np.ndarray, overlay: np.ndarray,
                               confidence_map: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
        """Create 2x2 panel (matches inference.py exactly)."""
        h, w = image_rgb.shape[:2]

        # Resize all to same size
        panel_h, panel_w = min(h, 512), min(w, 512)

        orig_resized = cv2.resize(image_rgb, (panel_w, panel_h))
        overlay_resized = cv2.resize(overlay, (panel_w, panel_h))
        conf_resized = cv2.resize(confidence_map, (panel_w, panel_h))

        # Class distribution chart
        chart = np.ones((panel_h, panel_w, 3), dtype=np.uint8) * 30
        unique, counts = np.unique(pred_mask, return_counts=True)

        active_ids = [c for c in range(1, self.num_classes) if c not in Config.EXCLUDED_CLASSES]
        num_active = len(active_ids)

        y_start = 30
        cv2.putText(chart, "Class Distribution", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        max_bar_w = panel_w - 120
        total_pixels = pred_mask.size

        for idx, cls_id in enumerate(active_ids):
            y = y_start + idx * (panel_h - 40) // max(num_active, 1)
            name = Config.CLASS_NAMES[cls_id][:12]
            color_rgb = Config.CLASS_COLORS_RGB.get(cls_id, [128, 128, 128])

            px_count = int((pred_mask == cls_id).sum())
            ratio = px_count / total_pixels
            bar_w = max(1, int(ratio * max_bar_w * 30))
            bar_w = min(bar_w, max_bar_w)

            cv2.rectangle(chart, (100, y - 5), (100 + bar_w, y + 12), color_rgb, -1)
            cv2.putText(chart, f"{name}", (5, y + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            cv2.putText(chart, f"{ratio*100:.1f}%", (105 + bar_w, y + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

        # Labels on panels
        for img, label in [(orig_resized, "Original"),
                          (overlay_resized, "Prediction Overlay"),
                          (conf_resized, "Confidence Map"),
                          (chart, "Class Distribution")]:
            cv2.putText(img, label, (5, img.shape[0] - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Assemble 2x2
        top = np.hstack([orig_resized, overlay_resized])
        bottom = np.hstack([conf_resized, chart])
        panel = np.vstack([top, bottom])

        return panel

    def _create_legend(self) -> np.ndarray:
        """Create color legend image (matches inference.py)."""
        active_ids = [c for c in range(1, self.num_classes) if c not in Config.EXCLUDED_CLASSES]
        row_h = 35
        legend_h = len(active_ids) * row_h + 20
        legend = np.ones((legend_h, 300, 3), dtype=np.uint8) * 30

        cv2.putText(legend, "CLASS LEGEND", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        for idx, cls_id in enumerate(active_ids):
            y = (idx + 1) * row_h + 5
            name = Config.CLASS_NAMES[cls_id]
            color_rgb = Config.CLASS_COLORS_RGB.get(cls_id, [128, 128, 128])

            cv2.rectangle(legend, (10, y - 15), (30, y + 5), color_rgb, -1)
            cv2.rectangle(legend, (10, y - 15), (30, y + 5), (200, 200, 200), 1)
            cv2.putText(legend, f"{cls_id}: {name}", (40, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        return legend

    # ── Main processing (from inference.py) ──────────────────────────

    def process_image(
        self,
        image_path: str,
        job_id: str,
        output_dir: str,
        use_tta: bool = False,
        enhance_contrast: bool = True,
        classes_to_show: Optional[List[int]] = None,
        confidence_threshold: float = 0.0,
        dilate: int = 0,
        include_visualizations: bool = True,
    ) -> Dict[str, Any]:
        """Process single image matching inference.py structure with Flask result format."""

        os.makedirs(output_dir, exist_ok=True)

        # Load image
        image = np.array(Image.open(image_path).convert("RGB"))
        filename = os.path.basename(image_path)
        basename = os.path.splitext(filename)[0]
        h, w = image.shape[:2]

        # Run inference
        prob_map = self.run_tiled_inference(image, use_tta=use_tta)

        if prob_map is None:
            return {
                'filename': filename,
                'status': 'failed',
                'error': 'Model not loaded',
            }

        # Get predictions
        pred_mask = np.argmax(prob_map, axis=0).astype(np.uint8)

        # Save original
        orig_path = os.path.join(output_dir, f"1_original.jpg")
        Image.fromarray(image).save(orig_path)

        # Save overlay
        overlay = self._create_colored_overlay(image, pred_mask)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        overlay_path = os.path.join(output_dir, f"2_prediction_overlay.jpg")
        cv2.imwrite(overlay_path, overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Save per-class masks
        class_masks = self._create_per_class_masks(image, pred_mask)
        per_class_list = []
        for cls_id, data in class_masks.items():
            view_bgr = cv2.cvtColor(data["view"], cv2.COLOR_RGB2BGR)
            cls_name = data["name"].replace(" ", "_")
            pc_path = os.path.join(output_dir, f"3_class_{cls_id:02d}_{cls_name}.jpg")
            cv2.imwrite(pc_path, view_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            per_class_list.append({
                'class': data['name'],
                'index': cls_id,
                'pixels': int(data['pixels']),
                'url': f'/api/files/{job_id}/3_class_{cls_id:02d}_{cls_name}.jpg',
            })

        # Save confidence heatmap
        conf_vis, confidence = self._create_confidence_heatmap(prob_map, image)
        conf_bgr = cv2.cvtColor(conf_vis, cv2.COLOR_RGB2BGR)
        conf_path = os.path.join(output_dir, f"4_confidence_map.jpg")
        cv2.imwrite(conf_path, conf_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Save raw mask
        mask_path = os.path.join(output_dir, f"5_raw_mask.png")
        Image.fromarray(pred_mask).save(mask_path)

        # Save summary panel
        panel = self._create_summary_panel(image, overlay, conf_vis, pred_mask)
        panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        panel_path = os.path.join(output_dir, f"6_summary_panel.jpg")
        cv2.imwrite(panel_path, panel_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Save legend
        legend = self._create_legend()
        legend_bgr = cv2.cvtColor(legend, cv2.COLOR_RGB2BGR)
        legend_path = os.path.join(output_dir, f"7_legend.jpg")
        cv2.imwrite(legend_path, legend_bgr)

        # Build report (matching inference.py)
        unique, counts = np.unique(pred_mask, return_counts=True)
        total = pred_mask.size

        # Build legend dict
        legend_list = []
        for cls_id in range(1, self.num_classes):
            if cls_id in Config.EXCLUDED_CLASSES:
                continue
            m = (pred_mask == cls_id)
            cnt = int(np.sum(m))
            if cnt > 0:
                name = Config.CLASS_NAMES[cls_id]
                color_rgb = Config.CLASS_COLORS_RGB.get(cls_id, [128, 128, 128])
                avg_conf = float(np.mean(confidence[m]))
                legend_list.append({
                    'class': name,
                    'color': '#{:02x}{:02x}{:02x}'.format(*color_rgb),
                    'count': cnt,
                    'avg_confidence': round(avg_conf, 3),
                })
        legend_list.sort(key=lambda x: x['count'], reverse=True)

        # Build confidence report
        per_class_conf = {}
        per_class_counts = {}
        total_conf = 0.0
        total_pixels = 0
        for cls_id in range(1, self.num_classes):
            if cls_id in Config.EXCLUDED_CLASSES:
                continue
            name = Config.CLASS_NAMES[cls_id]
            m = (pred_mask == cls_id)
            cnt = int(np.sum(m))
            per_class_counts[name] = cnt
            if cnt > 0:
                avg = float(np.mean(confidence[m]))
                per_class_conf[name] = round(avg, 3)
                total_conf += avg * cnt
                total_pixels += cnt
            else:
                per_class_conf[name] = 0.0

        overall_conf = round(total_conf / total_pixels, 3) if total_pixels > 0 else 0.0

        confidence_report = {
            'overall_confidence': overall_conf,
            'per_class': per_class_conf,
            'class_counts': per_class_counts,
        }

        # Build summary
        detected_classes = [l['class'] for l in legend_list]
        strongest = legend_list[0]['class'] if legend_list else None
        weakest = legend_list[-1]['class'] if legend_list else None
        summary = {
            'total_detected_lines': sum(l['count'] for l in legend_list),
            'total_classes_detected': len(detected_classes),
            'detected_classes': detected_classes,
            'strongest_class': strongest,
            'weakest_class': weakest,
            'quality_badge': (
                'good' if overall_conf > 0.7 else 'moderate' if overall_conf > 0.5 else 'review'
            ),
        }

        # Return result matching frontend expectations
        result = {
            'filename': filename,
            'status': 'success',
            'original_url': f'/api/files/{job_id}/1_original.jpg',
            'detection_result_url': f'/api/files/{job_id}/2_prediction_overlay.jpg',
            'legend': legend_list,
            'confidence_report': confidence_report,
            'summary': summary,
            'downloads': {
                'overlay_png': f'/api/download/{job_id}/overlay?filename=2_prediction_overlay.jpg',
                'raw_mask_png': f'/api/download/{job_id}/mask?filename=5_raw_mask.png',
            },
            'per_class_overlays': per_class_list,
            'visualizations': {
                'confidence_heatmap_url': f'/api/files/{job_id}/4_confidence_map.jpg',
                'summary_panel_url': f'/api/files/{job_id}/6_summary_panel.jpg',
            },
        }

        return result
