"""
Inference Engine for Engineering Drawing Line Detection
--------------------------------------------------------
Line-detection-only engine.  Supports:
  - Dynamic model checkpoint loading / swapping
  - Selective class filtering (show only chosen classes)
  - TTA toggle, confidence threshold, line dilation
  - Per-class overlay generation

The inference pipeline (tiling, temperature, post-processing) is
identical to inference_step3.py and test_inference.py.
"""

import os
import sys
import json
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

import torch

# Add parent dir for model imports
FLASK_DIR = Path(__file__).parent
APPROACH2_DIR = FLASK_DIR.parent
sys.path.insert(0, str(APPROACH2_DIR))
sys.path.insert(0, str(FLASK_DIR))

from config import Config
from inference.line_detection_inference import LineDetectionInference
from postprocess import create_debug_overlay


class InferenceEngine:
    """
    Line-detection inference engine.

    Features ported from test_inference.py:
      - Dynamic checkpoint selection
      - Class filtering (all 10 or specific subset)
      - TTA (test-time augmentation) toggle
      - Confidence threshold gating
      - Line dilation for visibility
      - Per-class overlay images
    """

    def __init__(
        self,
        line_detection_model_path: str = None,
        legend_path: str = None,
        num_classes: int = 11,
        backbone: str = "resnet50",
        device: str = "cuda",
        tile_size: int = 1024,
        overlap: int = 256,
        temperature: float = 0.5,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.backbone = backbone
        self.tile_size = tile_size
        self.overlap = overlap
        self.temperature = temperature
        self._current_checkpoint = None

        if line_detection_model_path is None:
            line_detection_model_path = Config.DEFAULT_CHECKPOINT
        if legend_path is None:
            legend_path = str(APPROACH2_DIR / "legend.json")

        self.legend_path = legend_path
        self.line_detection: Optional[LineDetectionInference] = None
        self._load_line_detection(line_detection_model_path)
        print(f"Inference Engine initialized on {self.device}")

    # ── model management ────────────────────────────────────────────

    def _load_line_detection(self, model_path: str):
        """Load (or reload) the line-detection model."""
        if model_path and os.path.exists(model_path):
            self.line_detection = LineDetectionInference(
                model_path=model_path,
                legend_path=self.legend_path,
                num_classes=self.num_classes,
                backbone=self.backbone,
                device=str(self.device),
                tile_size=self.tile_size,
                overlap=self.overlap,
                temperature=self.temperature,
                class_names=Config.CLASS_NAMES,
                class_colors=Config.CLASS_COLORS_RGB,
            )
            self._current_checkpoint = model_path
            print(f"Line detection model loaded: {model_path}")
        else:
            print(f"Warning: model not found at {model_path}")

    def switch_model(self, checkpoint_path: str) -> bool:
        """Hot-swap the model checkpoint (called from API)."""
        if checkpoint_path == self._current_checkpoint:
            return True
        if not os.path.exists(checkpoint_path):
            return False
        self._load_line_detection(checkpoint_path)
        return self.line_detection is not None

    @staticmethod
    def list_checkpoints() -> List[Dict]:
        return Config.list_checkpoints()

    # ── per-image processing ────────────────────────────────────────

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
    ) -> Dict[str, Any]:
        """
        Process a single image — line detection only.

        Args:
            image_path:           path to image
            job_id:               job identifier for URL generation
            output_dir:           where to save outputs
            use_tta:              test-time augmentation
            enhance_contrast:     adaptive CLAHE
            classes_to_show:      list of class indices (1-10), None = all
            confidence_threshold: min softmax confidence (0–1)
            dilate:               dilate lines by N px
        """
        os.makedirs(output_dir, exist_ok=True)

        image = np.array(Image.open(image_path).convert("RGB"))
        filename = os.path.basename(image_path)
        basename = os.path.splitext(filename)[0]

        result: Dict[str, Any] = {
            'filename': filename,
            'original_url': f'/api/files/{job_id}/original_{filename}',
            'detection_result_url': None,
            'legend': [],
            'confidence_report': {},
            'summary': {},
            'downloads': {},
            'per_class_overlays': [],
        }

        # save original
        Image.fromarray(image).save(os.path.join(output_dir, f'original_{filename}'))

        if self.line_detection is None:
            return result

        # determine classes (exclude phantom, break, hidden)
        Config._init_excluded_indices()
        excluded = Config.EXCLUDED_CLASS_INDICES
        all_classes = [c for c in range(1, self.num_classes) if c not in excluded]
        if classes_to_show is None or len(classes_to_show) == 0:
            classes_to_show = all_classes
        else:
            # Remove any explicitly-requested excluded classes
            classes_to_show = [c for c in classes_to_show if c not in excluded]

        # run inference
        det = self.line_detection.process_image(
            image_path=image_path,
            output_dir=output_dir,
            use_tta=use_tta,
            enhance_contrast=enhance_contrast,
            use_crf=False,
            use_bilateral=True,
            alpha=2.0,
            default_threshold=0.08,
            force_argmax=True,
            min_confidence=0.02,
        )

        if not det.get('success', False):
            return result

        det_image   = det['_image']
        argmax_mask = det['_argmax_mask']
        prob_class  = det['_prob_class']
        skeleton    = det['_skeleton']

        # ── zero out excluded classes in probability map ────────────
        for c in excluded:
            if c < prob_class.shape[0]:
                prob_class[c, :, :] = 0.0

        # Recompute argmax after zeroing excluded classes
        argmax_mask = np.argmax(prob_class, axis=0).astype(np.uint8)

        # ── apply class filter ──────────────────────────────────────
        mask_filtered = argmax_mask.copy()
        for c in all_classes:
            if c not in classes_to_show:
                mask_filtered[mask_filtered == c] = 0

        # ── apply confidence gating ─────────────────────────────────
        if confidence_threshold > 0:
            max_prob = np.max(prob_class, axis=0)
            mask_filtered[max_prob < confidence_threshold] = 0

        # ── apply dilation ──────────────────────────────────────────
        if dilate > 0:
            dilated = np.zeros_like(mask_filtered)
            kern = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1)
            )
            for c in classes_to_show:
                m = (mask_filtered == c).astype(np.uint8)
                if m.any():
                    m = cv2.dilate(m, kern)
                    dilated[m > 0] = c
            mask_filtered = dilated

        # ── build overlay (test_inference.py style) ─────────────────
        overlay = self._create_overlay(
            det_image, mask_filtered, classes_to_show, alpha=0.55
        )
        overlay_path = os.path.join(output_dir, f'{basename}_overlay.png')
        Image.fromarray(overlay).save(overlay_path)
        result['detection_result_url'] = f'/api/files/{job_id}/{basename}_overlay.png'

        # ── per-class overlays ──────────────────────────────────────
        per_class_dir = os.path.join(output_dir, f'{basename}_per_class')
        os.makedirs(per_class_dir, exist_ok=True)
        per_class_list = []
        for cid in classes_to_show:
            m = (mask_filtered == cid)
            if not m.any():
                continue
            pc_img = det_image.copy().astype(np.float32)
            col = Config.CLASS_COLORS_RGB.get(
                Config.CLASS_NAMES[cid], [255, 255, 255]
            )
            cl = np.zeros_like(det_image, dtype=np.float32)
            cl[m] = col
            pc_img[m] = 0.45 * pc_img[m] + 0.55 * cl[m]
            pc_img = np.clip(pc_img, 0, 255).astype(np.uint8)
            cname = Config.CLASS_NAMES[cid]
            fname = f'{basename}_class{cid}_{cname}.png'
            Image.fromarray(pc_img).save(os.path.join(per_class_dir, fname))
            per_class_list.append({
                'class': cname,
                'index': cid,
                'url': f'/api/files/{job_id}/{basename}_per_class/{fname}',
            })
        result['per_class_overlays'] = per_class_list

        # ── legend + confidence + summary ───────────────────────────
        result['legend'] = self._generate_legend(
            mask_filtered, prob_class, classes_to_show
        )
        result['confidence_report'] = self._compute_confidence(
            prob_class, mask_filtered, classes_to_show
        )

        detected_classes = [l['class'] for l in result['legend']]
        strongest = result['legend'][0]['class'] if result['legend'] else None
        weakest = result['legend'][-1]['class'] if result['legend'] else None
        overall = result['confidence_report'].get('overall_confidence', 0)

        result['summary'] = {
            'total_detected_lines': sum(l['count'] for l in result['legend']),
            'total_classes_detected': len(detected_classes),
            'detected_classes': detected_classes,
            'strongest_class': strongest,
            'weakest_class': weakest,
            'quality_badge': (
                'good' if overall > 0.7 else 'moderate' if overall > 0.5 else 'review'
            ),
        }

        result['stats'] = det.get('stats', {})
        result['downloads']['overlay_png'] = (
            f'/api/download/{job_id}/overlay?filename={basename}_overlay.png'
        )
        result['downloads']['mask_png'] = (
            f'/api/download/{job_id}/mask?filename=masks/{basename}_mask.png'
        )

        # save JSON report
        report = {
            'confidence_report': result['confidence_report'],
            'legend': result['legend'],
            'summary': result['summary'],
        }
        rpath = os.path.join(output_dir, f'{basename}_report.json')
        with open(rpath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        result['downloads']['confidence_json'] = (
            f'/api/download/{job_id}/report?filename={basename}_report.json'
        )
        result['downloads']['all_zip'] = f'/api/download/{job_id}/all_zip'

        return result

    # ── helpers ──────────────────────────────────────────────────────

    def _create_overlay(
        self,
        image: np.ndarray,
        prediction: np.ndarray,
        classes_to_show: List[int],
        alpha: float = 0.55,
    ) -> np.ndarray:
        """Colour-coded overlay with embedded legend panel (test_inference style)."""
        h, w = image.shape[:2]
        overlay = image.copy().astype(np.float32)
        cmask = np.zeros_like(image, dtype=np.float32)
        fg = np.zeros((h, w), dtype=bool)

        class_names = Config.CLASS_NAMES
        class_colors = Config.CLASS_COLORS_RGB

        for cid in classes_to_show:
            m = (prediction == cid)
            if m.any():
                col = class_colors.get(class_names[cid], [255, 255, 255])
                cmask[m] = col
                fg |= m

        overlay[fg] = (1 - alpha) * overlay[fg] + alpha * cmask[fg]
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # legend panel
        lw = 280
        rh = 32
        pad = 15
        panel_h = max(h, pad * 2 + rh * len(classes_to_show) + 50)
        panel = np.full((panel_h, lw, 3), 30, dtype=np.uint8)
        cv2.putText(panel, "DETECTED LINES", (pad, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.line(panel, (pad, 40), (lw - pad, 40), (100, 100, 100), 1)

        yo = 60
        for cid in classes_to_show:
            if cid in Config.EXCLUDED_CLASS_INDICES:
                continue
            name = class_names[cid] if cid < len(class_names) else f"class_{cid}"
            col = tuple(class_colors.get(name, [255, 255, 255]))
            cnt = int(np.sum(prediction == cid))
            cv2.rectangle(panel, (pad, yo - 12), (pad + 20, yo + 4), col, -1)
            cv2.rectangle(panel, (pad, yo - 12), (pad + 20, yo + 4), (200, 200, 200), 1)
            st = f"{cnt:,}px" if cnt > 0 else "not detected"
            tc = col if cnt > 0 else (100, 100, 100)
            cv2.putText(panel, name, (pad + 28, yo),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, tc, 1)
            cv2.putText(panel, f"  {st}", (pad + 28, yo + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
            yo += rh

        result_h = max(h, panel.shape[0])
        result = np.full((result_h, w + lw, 3), 30, dtype=np.uint8)
        result[:h, :w] = overlay
        result[:panel.shape[0], w:] = panel[:result_h]
        return result

    def _compute_confidence(
        self,
        prob_class: np.ndarray,
        seg_mask: np.ndarray,
        classes_to_show: List[int],
    ) -> Dict[str, Any]:
        class_names = Config.CLASS_NAMES
        per_class = {}
        counts = {}
        total_conf = 0.0
        total_pixels = 0

        for cid in classes_to_show:
            if cid in Config.EXCLUDED_CLASS_INDICES:
                continue
            name = class_names[cid] if cid < len(class_names) else f"class_{cid}"
            m = (seg_mask == cid)
            cnt = int(np.sum(m))
            counts[name] = cnt
            if cnt > 0:
                avg = float(np.mean(prob_class[cid][m]))
                per_class[name] = round(avg, 3)
                total_conf += avg * cnt
                total_pixels += cnt
            else:
                per_class[name] = 0.0

        overall = round(total_conf / total_pixels, 3) if total_pixels > 0 else 0.0
        low = [c for c, v in per_class.items() if 0 < v < 0.5]
        missing = [c for c, v in counts.items() if v == 0]

        return {
            'overall_confidence': overall,
            'per_class': per_class,
            'class_counts': counts,
            'low_confidence_items': len(low),
            'low_confidence_classes': low,
            'missing_classes': missing,
        }

    def _generate_legend(
        self,
        seg_mask: np.ndarray,
        prob_class: np.ndarray,
        classes_to_show: List[int],
    ) -> List[Dict]:
        class_names = Config.CLASS_NAMES
        class_colors = Config.CLASS_COLORS_RGB
        legend = []

        for cid in classes_to_show:
            if cid in Config.EXCLUDED_CLASS_INDICES:
                continue
            name = class_names[cid] if cid < len(class_names) else f"class_{cid}"
            m = (seg_mask == cid)
            cnt = int(np.sum(m))
            if cnt > 0:
                avg = float(np.mean(prob_class[cid][m]))
                col = class_colors.get(name, [128, 128, 128])
                legend.append({
                    'class': name,
                    'color': '#{:02x}{:02x}{:02x}'.format(*col),
                    'count': cnt,
                    'avg_confidence': round(avg, 3),
                })

        legend.sort(key=lambda x: x['count'], reverse=True)
        return legend
