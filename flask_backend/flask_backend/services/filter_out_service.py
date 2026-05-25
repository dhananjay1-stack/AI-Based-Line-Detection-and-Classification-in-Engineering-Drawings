import os
import sys
import time
from pathlib import Path
from typing import Dict, Any
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.filter_out_inference import FilterOutInference
from utils.logging_utils import (
    log_job_event, log_processing_time, log_inference, log_output_saved
)


class FilterOutService:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        max_imgsz: int = 2048
    ):
       
        self.inference = FilterOutInference(
            model_path=model_path,
            conf_threshold=conf_threshold,
            max_imgsz=max_imgsz
        )

    def is_available(self) -> bool:
        """Check if the model is loaded and ready."""
        return self.inference.model is not None

    def process_single_image(
        self,
        image_path: str,
        output_dir: str,
        job_id: str,
        save_debug: bool = True
    ) -> Dict[str, Any]:
        start_time = time.time()
        filename = os.path.basename(image_path)
        basename = os.path.splitext(filename)[0]

        log_job_event(job_id, 'filter_out_start', {'filename': filename})
        image = np.array(Image.open(image_path).convert('RGB'))
        filtered_dir = os.path.join(output_dir, 'filtered_images')
        overlays_dir = os.path.join(output_dir, 'debug_overlays')

        os.makedirs(filtered_dir, exist_ok=True)
        if save_debug:
            os.makedirs(overlays_dir, exist_ok=True)

        inference_start = time.time()
        filter_results = self.inference.run_inference(image)
        inference_time = time.time() - inference_start

        log_inference(job_id, 'filter_out', filename, inference_time)

        filtered_image = self.inference.apply_filter(image, filter_results['filter_mask'])

        filtered_path = os.path.join(filtered_dir, f'{basename}_filtered.png')
        Image.fromarray(filtered_image).save(filtered_path)
        log_output_saved(job_id, 'filtered_image', filtered_path)

        mask_path = os.path.join(filtered_dir, f'{basename}_filter_mask.png')
        Image.fromarray(filter_results['filter_mask']).save(mask_path)
        log_output_saved(job_id, 'filter_mask', mask_path)

        overlay_path = None
        if save_debug:
            overlay = self.inference.create_debug_overlay(image, filter_results['filter_mask'])
            overlay_path = os.path.join(overlays_dir, f'{basename}_filter_overlay.png')
            Image.fromarray(overlay).save(overlay_path)
            log_output_saved(job_id, 'filter_overlay', overlay_path)

        # Compute statistics
        statistics = self.inference.compute_statistics(filter_results)

        processing_time = time.time() - start_time
        log_processing_time(job_id, 'filter_out_total', processing_time, filename)

        result = {
            'filename': filename,
            'success': True,
            'filtered_image_path': f'/api/files/{job_id}/filtered_images/{basename}_filtered.png',
            'filter_mask_path': f'/api/files/{job_id}/filtered_images/{basename}_filter_mask.png',
            'confidence_score': statistics['overall_filter_score'],
            'removed_count': statistics['removed_pixel_count'],
            'retained_count': statistics['retained_pixel_count'],
            'removal_percentage': statistics['removal_percentage'],
            'detection_count': statistics['detection_count'],
            'timings': {
                'inference': round(inference_time, 2),
                'total': round(processing_time, 2)
            },
            '_filtered_image': filtered_image,
            '_filter_mask': filter_results['filter_mask']
        }

        if overlay_path:
            result['debug_overlay_path'] = f'/api/files/{job_id}/debug_overlays/{basename}_filter_overlay.png'

        return result

    def get_filtered_image(self, result: Dict[str, Any]) -> np.ndarray:
        return result.get('_filtered_image')
