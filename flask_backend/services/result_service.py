import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional


class ResultService:
    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def format_api_result(
        job_id: str,
        mode: str,
        results: List[Dict[str, Any]],
        original_url_prefix: str = '/api/files'
    ) -> List[Dict[str, Any]]:
        formatted = []

        for result in results:
            filename = result.get('filename', '')
            basename = os.path.splitext(filename)[0]

            item = {
                'filename': filename,
                'original_url': f'{original_url_prefix}/{job_id}/input/{filename}',
                'detection_result_url': None,
                'filtered_result_url': None,
                'legend': [],
                'confidence_report': {},
                'filter_report': {'enabled': False},
                'summary': {},
                'downloads': {}
            }

            # Line detection results
            if 'line_detection' in result:
                ld = result['line_detection']
                item['detection_result_url'] = ld.get('overlay_path')
                item['legend'] = ld.get('legend', [])

                item['confidence_report'] = {
                    'overall_confidence': ld.get('confidence_score', 0),
                    'per_class': ld.get('class_confidences', {}),
                    'class_counts': ld.get('class_counts', {}),
                    'low_confidence_classes': [
                        c for c, conf in ld.get('class_confidences', {}).items()
                        if 0 < conf < 0.5
                    ]
                }

                detected_classes = [l['class'] for l in ld.get('legend', [])]
                total_lines = sum(l['count'] for l in ld.get('legend', []))
                strongest = detected_classes[0] if detected_classes else None
                weakest = detected_classes[-1] if detected_classes else None

                item['summary'] = {
                    'total_detected_lines': total_lines,
                    'total_classes_detected': len(detected_classes),
                    'detected_classes': detected_classes,
                    'strongest_class': strongest,
                    'weakest_class': weakest,
                    'quality_badge': (
                        'good' if ld.get('confidence_score', 0) > 0.7 else
                        'moderate' if ld.get('confidence_score', 0) > 0.5 else
                        'review'
                    )
                }

                item['downloads']['overlay_png'] = f'/api/download/{job_id}/overlay?filename={basename}_overlay.png'
                item['downloads']['mask_png'] = f'/api/download/{job_id}/mask?filename={basename}_mask.png'

            # Filter-out results
            if 'filter_out' in result:
                fo = result['filter_out']
                item['filtered_result_url'] = fo.get('filtered_image_path')

                item['filter_report'] = {
                    'enabled': True,
                    'overall_filter_score': round(fo.get('confidence_score', 0), 3),
                    'removed_non_essential_count': fo.get('removed_count', 0),
                    'retained_essential_count': fo.get('retained_count', 0),
                    'removal_percentage': fo.get('removal_percentage', 0),
                    'notes': (
                        'Non-essential elements removed successfully'
                        if fo.get('removed_count', 0) > 0 else
                        'No non-essential elements detected'
                    )
                }

                item['downloads']['filtered_png'] = f'/api/download/{job_id}/filtered?filename={basename}_filtered.png'
                item['downloads']['filter_mask_png'] = f'/api/download/{job_id}/filter_mask?filename={basename}_filter_mask.png'

            item['downloads']['confidence_json'] = f'/api/download/{job_id}/report?filename={basename}_report.json'
            item['downloads']['all_zip'] = f'/api/download/{job_id}/all_zip'

            formatted.append(item)

        return formatted

    @staticmethod
    def save_processing_stats(
        output_dir: str,
        stats: Dict[str, Any]
    ) -> str:
       
        path = os.path.join(output_dir, 'processing_stats.json')
        with open(path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        return path

    @staticmethod
    def save_result_json(
        output_dir: str,
        job_id: str,
        mode: str,
        status: str,
        results: List[Dict],
        warnings: List[str] = None
    ) -> str:
        data = {
            'job_id': job_id,
            'mode': mode,
            'status': status,
            'results': results,
            'warnings': warnings or [],
            'generated_at': datetime.now().isoformat()
        }

        path = os.path.join(output_dir, 'result.json')
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return path

    @staticmethod
    def save_image_report(
        output_dir: str,
        basename: str,
        result: Dict[str, Any]
    ) -> str:
        path = os.path.join(output_dir, f'{basename}_report.json')

        report = {
            'filename': result.get('filename'),
            'processing_time': result.get('processing_time', 0)
        }

        if 'line_detection' in result:
            ld = result['line_detection']
            report['confidence_report'] = {
                'overall_confidence': ld.get('confidence_score', 0),
                'per_class': ld.get('class_confidences', {}),
                'class_counts': ld.get('class_counts', {})
            }
            report['legend'] = ld.get('legend', [])

        if 'filter_out' in result:
            fo = result['filter_out']
            report['filter_report'] = {
                'confidence_score': fo.get('confidence_score', 0),
                'removed_count': fo.get('removed_count', 0),
                'retained_count': fo.get('retained_count', 0)
            }

        with open(path, 'w') as f:
            json.dump(report, f, indent=2)

        return path
