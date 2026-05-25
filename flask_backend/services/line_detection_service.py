import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.line_detection_inference import LineDetectionInference
from utils.logging_utils import (
    log_job_event, log_processing_time, log_inference, log_output_saved
)


class LineDetectionService:
    """
    Line Detection Service that delegates entirely to
    LineDetectionInference.process_image() to match inference_step3.py.

    Deliverables (matching inference_step3.py defaults):
    - masks/{basename}_mask.png
    - binary_masks/{basename}_class{N}.png
    - processing_stats.json
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
        temperature: float = 0.5,
        class_names: List[str] = None,
        class_colors: Dict[str, List[int]] = None
    ):
        self.inference = LineDetectionInference(
            model_path=model_path,
            legend_path=legend_path,
            num_classes=num_classes,
            backbone=backbone,
            device=device,
            tile_size=tile_size,
            overlap=overlap,
            temperature=temperature,
            class_names=class_names,
            class_colors=class_colors
        )

        self.class_names = self.inference.class_names
        self.class_colors = self.inference.class_colors

    def is_available(self) -> bool:
        return self.inference.model is not None

    def process_single_image(
        self,
        image_path: str,
        output_dir: str,
        job_id: str,
        use_tta: bool = False,
        enhance_contrast: bool = True,
        image_override: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Process a single image by delegating to
        LineDetectionInference.process_image().

        This produces the exact same deliverables as inference_step3.py:
        - masks/{basename}_mask.png
        - binary_masks/{basename}_class{N}.png
        """
        start_time = time.time()
        filename = os.path.basename(image_path)
        basename = os.path.splitext(filename)[0]

        log_job_event(job_id, 'line_detection_start', {'filename': filename})

        # If an image override is provided, save it as a temp file
        # so process_image() can load it (it expects a file path)
        actual_image_path = image_path
        if image_override is not None:
            temp_path = os.path.join(output_dir, f'_temp_{filename}')
            os.makedirs(output_dir, exist_ok=True)
            Image.fromarray(image_override).save(temp_path)
            actual_image_path = temp_path

        # Delegate to the inference pipeline (matches inference_step3.py)
        result = self.inference.process_image(
            image_path=actual_image_path,
            output_dir=output_dir,
            use_tta=use_tta,
            enhance_contrast=enhance_contrast,
            use_crf=False,
            use_bilateral=True,
            alpha=2.0,
            default_threshold=0.08,
            force_argmax=True,
            min_confidence=0.02
        )

        # Clean up temp file if created
        if image_override is not None and os.path.exists(temp_path):
            os.remove(temp_path)

        if not result.get('success', False):
            return {
                'filename': filename,
                'success': False,
                'error': result.get('error', 'Inference failed'),
                'processing_time': time.time() - start_time
            }

        processing_time = time.time() - start_time
        log_processing_time(job_id, 'line_detection_total', processing_time, filename)
        log_output_saved(job_id, 'mask', result['paths']['mask'])

        # Return API-accessible paths and stats
        return {
            'filename': filename,
            'success': True,
            'mask_path': f'/api/files/{job_id}/masks/{basename}_mask.png',
            'binary_masks_path': f'/api/files/{job_id}/binary_masks/',
            'stats': result.get('stats', {}),
            'timings': result.get('timings', {})
        }

    def generate_hex_colors(self) -> Dict[str, str]:
        """Get class colors as hex strings."""
        hex_colors = {}
        for class_name, rgb in self.class_colors.items():
            hex_colors[class_name] = '#{:02x}{:02x}{:02x}'.format(*rgb)
        return hex_colors
