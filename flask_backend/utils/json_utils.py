import json
from datetime import datetime
from typing import Dict, List, Any, Optional


def load_json(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Dict, path: str, indent: int = 2) -> str:
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)
    return path


def create_processing_stats(
    job_id: str,
    mode: str,
    images: List[Dict[str, Any]],
    total_time: float
) -> Dict[str, Any]:
    return {
        'job_id': job_id,
        'mode': mode,
        'processed_at': datetime.now().isoformat(),
        'total_images': len(images),
        'total_processing_time': round(total_time, 2),
        'average_time_per_image': round(total_time / len(images), 2) if images else 0,
        'images': images
    }


def create_result_json(
    job_id: str,
    mode: str,
    status: str,
    results: List[Dict[str, Any]],
    warnings: List[str] = None,
    processing_time: float = 0
) -> Dict[str, Any]:
    return {
        'job_id': job_id,
        'mode': mode,
        'status': status,
        'results': results,
        'warnings': warnings or [],
        'processing_time': round(processing_time, 2),
        'generated_at': datetime.now().isoformat()
    }


def create_image_result(
    filename: str,
    line_detection: Optional[Dict[str, Any]] = None,
    filter_out: Optional[Dict[str, Any]] = None,
    processing_time: float = 0,
    warnings: List[str] = None
) -> Dict[str, Any]:
    result = {
        'filename': filename,
        'processing_time': round(processing_time, 2),
        'warnings': warnings or []
    }

    if line_detection:
        result['line_detection'] = line_detection

    if filter_out:
        result['filter_out'] = filter_out

    return result


def create_line_detection_result(
    overlay_path: str,
    binary_mask_path: str,
    mask_path: str,
    vector_path: Optional[str] = None,
    confidence_score: float = 0,
    class_counts: Dict[str, int] = None,
    class_confidences: Dict[str, float] = None
) -> Dict[str, Any]:
    return {
        'overlay_path': overlay_path,
        'binary_mask_path': binary_mask_path,
        'mask_path': mask_path,
        'vector_path': vector_path,
        'confidence_score': round(confidence_score, 3),
        'class_counts': class_counts or {},
        'class_confidences': class_confidences or {}
    }


def create_filter_out_result(
    filtered_image_path: str,
    filter_mask_path: str,
    confidence_score: float = 0,
    removed_count: int = 0,
    retained_count: int = 0
) -> Dict[str, Any]:
    return {
        'filtered_image_path': filtered_image_path,
        'filter_mask_path': filter_mask_path,
        'confidence_score': round(confidence_score, 3),
        'removed_count': removed_count,
        'retained_count': retained_count
    }
