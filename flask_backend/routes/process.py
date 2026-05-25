import os
import sys
import time
import threading
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Blueprint, request, jsonify
from PIL import Image
import numpy as np
import shutil

from utils.logging_utils import (
    log_processing_start, log_processing_complete, log_processing_failed
)
from services.result_service import ResultService


def create_process_routes(
    job_service,
    line_detection_service,
    filter_out_service,
    config
):
    """
    Create process routes blueprint.

    Args:
        job_service: JobService instance
        line_detection_service: LineDetectionService instance
        filter_out_service: FilterOutService instance
        config: Application config

    Returns:
        Flask Blueprint
    """
    bp = Blueprint('process', __name__)

    def process_job(job_id: str, files: list, mode: str):
        """Background job processor."""
        start_time = time.time()

        try:
            job_service.update_status(job_id, job_service.STATUS_PROCESSING)
            log_processing_start(job_id, mode, len(files))

            # Create output directories
            output_dir = job_service.get_output_dir(job_id)
            dirs = job_service.create_job_directories(job_id, mode)

            results = []
            image_stats = []

            for i, file_path in enumerate(files):
                filename = os.path.basename(file_path)
                basename = os.path.splitext(filename)[0]

                job_service.update_progress(job_id, i + 1, len(files), filename)

                img_start_time = time.time()
                result = {'filename': filename}

                # Copy original to input directory
                input_copy = os.path.join(dirs['input'], filename)
                shutil.copy2(file_path, input_copy)

                # Current image for processing
                current_image = None

                # Filter-out mode
                if mode in ['filter', 'both']:
                    filter_result = filter_out_service.process_single_image(
                        image_path=file_path,
                        output_dir=output_dir,
                        job_id=job_id
                    )

                    result['filter_out'] = {
                        'filtered_image_path': filter_result.get('filtered_image_path'),
                        'filter_mask_path': filter_result.get('filter_mask_path'),
                        'confidence_score': filter_result.get('confidence_score', 0),
                        'removed_count': filter_result.get('removed_count', 0),
                        'retained_count': filter_result.get('retained_count', 0),
                        'removal_percentage': filter_result.get('removal_percentage', 0),
                        'timings': filter_result.get('timings', {})
                    }

                    # Get filtered image for combined mode
                    if mode == 'both':
                        current_image = filter_out_service.get_filtered_image(filter_result)

                # Line detection mode
                if mode in ['detection', 'both']:
                    detection_result = line_detection_service.process_single_image(
                        image_path=file_path,
                        output_dir=output_dir,
                        job_id=job_id,
                        use_tta=config.USE_TTA,
                        image_override=current_image  # Use filtered image if available
                    )

                    result['line_detection'] = {
                        'overlay_path': detection_result.get('overlay_path'),
                        'mask_path': detection_result.get('mask_path'),
                        'binary_masks_path': detection_result.get('binary_masks_path'),
                        'vectors_path': detection_result.get('vectors_path'),
                        'confidence_score': detection_result.get('confidence_score', 0),
                        'class_counts': detection_result.get('class_counts', {}),
                        'class_confidences': detection_result.get('class_confidences', {}),
                        'legend': detection_result.get('legend', []),
                        'timings': detection_result.get('timings', {})
                    }

                img_time = time.time() - img_start_time
                result['processing_time'] = round(img_time, 2)

                results.append(result)

                # Save per-image report
                ResultService.save_image_report(output_dir, basename, result)

                # Collect stats for processing_stats.json
                image_stats.append({
                    'filename': filename,
                    'processing_time': round(img_time, 2),
                    'success': True
                })

            total_time = time.time() - start_time

            # Format results for API
            formatted_results = ResultService.format_api_result(job_id, mode, results)

            # Save processing_stats.json
            stats = ResultService.create_processing_stats(
                job_id=job_id,
                mode=mode,
                images=image_stats,
                total_time=total_time
            )
            ResultService.save_processing_stats(output_dir, stats)

            # Save result.json
            ResultService.save_result_json(
                output_dir=output_dir,
                job_id=job_id,
                mode=mode,
                status='completed',
                results=results
            )

            # Update job with results
            job_service.set_results(job_id, formatted_results)
            job_service.update_status(job_id, job_service.STATUS_COMPLETED)

            log_processing_complete(job_id, total_time)

        except Exception as e:
            error_msg = str(e)
            job_service.update_status(job_id, job_service.STATUS_FAILED, error_msg)
            log_processing_failed(job_id, error_msg)
            import traceback
            traceback.print_exc()

    @bp.route('/api/run', methods=['POST'])
    def run_processing():
        """
        Start processing job.

        POST /api/run
        - JSON body: { job_id: string, mode: "detection"|"filter"|"both" }

        Returns:
            - job_id: Job identifier
            - status: "queued"
            - mode: Processing mode
        """
        data = request.get_json()

        if not data or 'job_id' not in data:
            return jsonify({'error': 'job_id required'}), 400

        job_id = data['job_id']
        mode = data.get('mode', 'detection')

        # Validate mode
        valid_modes = ['detection', 'filter', 'both']
        if mode not in valid_modes:
            return jsonify({
                'error': f'Invalid mode. Use: {", ".join(valid_modes)}'
            }), 400

        # Check job exists and can be processed
        job = job_service.get_job(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404

        if not job_service.can_process(job_id):
            return jsonify({'error': 'Job already processing'}), 400

        # Check model availability
        if mode in ['detection', 'both'] and not line_detection_service.is_available():
            return jsonify({
                'error': 'Line detection model not available'
            }), 503

        if mode in ['filter', 'both'] and not filter_out_service.is_available():
            return jsonify({
                'error': 'Filter model not available'
            }), 503

        # Update job
        job_service.set_mode(job_id, mode)
        job_service.update_status(job_id, job_service.STATUS_QUEUED)

        # Create output directory
        output_dir = job_service.get_output_dir(job_id)
        os.makedirs(output_dir, exist_ok=True)

        # Start background processing
        thread = threading.Thread(
            target=process_job,
            args=(job_id, job['files'], mode)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'mode': mode
        })

    @bp.route('/api/status/<job_id>', methods=['GET'])
    def get_status(job_id):
        """
        Get job status.

        GET /api/status/<job_id>

        Returns:
            Job status including progress if processing
        """
        job = job_service.get_job(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404

        # Don't include full file paths in response
        response = {
            'job_id': job['job_id'],
            'status': job['status'],
            'mode': job.get('mode'),
            'created_at': job.get('created_at'),
            'progress': job.get('progress'),
            'error': job.get('error')
        }

        return jsonify(response)

    return bp
