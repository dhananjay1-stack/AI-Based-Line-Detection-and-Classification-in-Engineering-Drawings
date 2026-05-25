"""
Filter-Out Inference Module (FIXED)
------------------------------------
Detects and removes non-essential parts of engineering drawings
using a YOLO segmentation model.

This module has been verified to match the backend filter model behavior.

Produces:
- filtered_images/
- debug_overlays/
- processing_stats.json
"""

import os
import sys
import time
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Add parent directory for imports
APPROACH2_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(APPROACH2_DIR))


class FilterOutInference:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        max_imgsz: int = 2048
    ):
        """
        Initialize filter-out inference.

        Args:
            model_path: Path to YOLO segmentation model
            conf_threshold: Confidence threshold for filtering
            max_imgsz: Maximum image size for YOLO inference
        """
        self.conf_threshold = conf_threshold
        self.max_imgsz = max_imgsz
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load the YOLO segmentation model."""
        if not os.path.exists(model_path):
            print(f"Warning: Filter model not found at {model_path}")
            return None

        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            print(f"Filter model loaded from {model_path}")
            return model
        except ImportError:
            print("Warning: ultralytics not installed. Filter mode disabled.")
            return None

    def run_inference(
        self,
        image: np.ndarray,
        conf_thresh: float = None
    ) -> Dict[str, Any]:
        """
        YOLO filter model to detect non-essential regions.

        Args:
            image: RGB image (H, W, 3)
            conf_thresh: Override default confidence threshold

        Returns:
            Dictionary with:
                - filter_probs: (H, W) probability map
                - filter_mask: (H, W) binary mask
                - filtered_count: Number of filtered pixels
                - retained_count: Number of retained pixels
                - filter_score: 1 - (filtered / total)
                - detections: List of detection info
        """
        h, w = image.shape[:2]
        conf_thresh = conf_thresh or self.conf_threshold

        filter_probs = np.zeros((h, w), dtype=np.float32)
        filter_mask = np.zeros((h, w), dtype=np.uint8)
        detections = []

        if self.model is None:
            return {
                'filter_probs': filter_probs,
                'filter_mask': filter_mask,
                'filtered_count': 0,
                'retained_count': h * w,
                'filter_score': 1.0,
                'detections': []
            }

        max_dim = max(h, w)
        imgsz = min(max_dim, self.max_imgsz)

        if max_dim > self.max_imgsz:
            scale = self.max_imgsz / max_dim
            new_h, new_w = int(h * scale), int(w * scale)
            resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized_img = image
            scale = 1.0

        results = self.model.predict(resized_img, imgsz=imgsz, conf=0.25, verbose=False)

        if results and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            if masks.shape[1:] != (h, w):
                resized_masks = []
                for m in masks:
                    resized_masks.append(cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR))
                masks = np.array(resized_masks)

            # Combine masks
            for i, m in enumerate(masks):
                filter_probs = np.maximum(filter_probs, m * confs[i])
                detections.append({
                    'class_id': int(classes[i]),
                    'confidence': float(confs[i]),
                    'area': int(np.sum(m > 0.5))
                })

        # Apply threshold
        filter_mask = (filter_probs > conf_thresh).astype(np.uint8) * 255
        filtered_count = int(np.sum(filter_mask > 0))
        retained_count = h * w - filtered_count
        filter_score = 1.0 - (filtered_count / (h * w)) if h * w > 0 else 1.0

        return {
            'filter_probs': filter_probs,
            'filter_mask': filter_mask,
            'filtered_count': filtered_count,
            'retained_count': retained_count,
            'filter_score': filter_score,
            'detections': detections
        }

    def apply_filter(
        self,
        image: np.ndarray,
        filter_mask: np.ndarray,
        fill_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """
        Apply filter mask to image (replace filtered regions).

        Args:
            image: RGB image
            filter_mask: Binary mask of regions to remove
            fill_color: Color to fill removed regions

        Returns:
            Filtered image
        """
        filtered_image = image.copy()
        filtered_image[filter_mask > 0] = fill_color
        return filtered_image

    def create_debug_overlay(
        self,
        image: np.ndarray,
        filter_mask: np.ndarray,
        alpha: float = 0.5,
        removed_color: Tuple[int, int, int] = (255, 0, 0),
        retained_color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        Debug overlay showing removed vs retained regions.

        Args:
            image: Original image
            filter_mask: Binary filter mask
            alpha: Overlay transparency
            removed_color: Color for removed regions
            retained_color: Color for retained regions

        Returns:
            Debug overlay image
        """
        overlay = image.copy().astype(np.float32)

        # colored overlay
        colored = np.zeros_like(image, dtype=np.float32)
        colored[filter_mask > 0] = removed_color
        colored[filter_mask == 0] = retained_color

        content_mask = np.any(image < 250, axis=2)
        blend_mask = content_mask

        overlay[blend_mask] = (
            (1 - alpha) * overlay[blend_mask] +
            alpha * colored[blend_mask]
        )

        return overlay.astype(np.uint8)

    def compute_statistics(
        self,
        filter_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute filter statistics.

        Args:
            filter_results: Results from run_inference

        Returns:
            Statistics dictionary
        """
        return {
            'overall_filter_score': round(filter_results['filter_score'], 3),
            'removed_pixel_count': filter_results['filtered_count'],
            'retained_pixel_count': filter_results['retained_count'],
            'removal_percentage': round(
                100 * filter_results['filtered_count'] /
                (filter_results['filtered_count'] + filter_results['retained_count']),
                2
            ) if (filter_results['filtered_count'] + filter_results['retained_count']) > 0 else 0,
            'detection_count': len(filter_results.get('detections', [])),
            'detections': filter_results.get('detections', [])
        }

    def process_image(
        self,
        image_path: str,
        output_dir: str,
        save_debug: bool = True
    ) -> Dict[str, Any]:
        start_time = time.time()

        image = np.array(Image.open(image_path).convert('RGB'))
        filename = os.path.basename(image_path)
        basename = os.path.splitext(filename)[0]

        filtered_dir = os.path.join(output_dir, 'filtered_images')
        overlays_dir = os.path.join(output_dir, 'debug_overlays')

        os.makedirs(filtered_dir, exist_ok=True)
        if save_debug:
            os.makedirs(overlays_dir, exist_ok=True)

        # Run inference
        inference_start = time.time()
        filter_results = self.run_inference(image)
        inference_time = time.time() - inference_start

        # Apply filter
        filtered_image = self.apply_filter(image, filter_results['filter_mask'])

        # Save outputs
        # 1. Filtered image
        filtered_path = os.path.join(filtered_dir, f'{basename}_filtered.png')
        Image.fromarray(filtered_image).save(filtered_path)

        # 2. Filter mask
        mask_path = os.path.join(filtered_dir, f'{basename}_filter_mask.png')
        Image.fromarray(filter_results['filter_mask']).save(mask_path)

        # 3. Debug overlay
        overlay_path = None
        if save_debug:
            overlay = self.create_debug_overlay(image, filter_results['filter_mask'])
            overlay_path = os.path.join(overlays_dir, f'{basename}_filter_overlay.png')
            Image.fromarray(overlay).save(overlay_path)

        # Compute statistics
        statistics = self.compute_statistics(filter_results)

        processing_time = time.time() - start_time

        return {
            'filename': filename,
            'success': True,
            'paths': {
                'filtered_image': filtered_path,
                'filter_mask': mask_path,
                'debug_overlay': overlay_path
            },
            'filtered_image': filtered_image,
            'filter_mask': filter_results['filter_mask'],
            'statistics': statistics,
            'timings': {
                'inference': round(inference_time, 2),
                'total': round(processing_time, 2)
            }
        }
