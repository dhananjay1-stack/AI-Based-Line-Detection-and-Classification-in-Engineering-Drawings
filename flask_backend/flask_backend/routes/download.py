import os
import zipfile
from flask import Blueprint, request, jsonify, send_file


def create_download_routes(job_service, config):
    bp = Blueprint('download', __name__)

    @bp.route('/api/download/<job_id>/<file_type>', methods=['GET'])
    def download_file(job_id, file_type):
        output_dir = os.path.join(config.OUTPUT_FOLDER, job_id)

        if not os.path.exists(output_dir):
            return jsonify({'error': 'Output not found'}), 404

        filename = request.args.get('filename', '')

        if file_type == 'all_zip':
            zip_path = os.path.join(output_dir, 'all_outputs.zip')
            if not os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(output_dir):
                        for file in files:
                            if file != 'all_outputs.zip':
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, output_dir)
                                zipf.write(file_path, arcname)

            return send_file(
                zip_path,
                as_attachment=True,
                download_name=f'{job_id}_outputs.zip'
            )

        # Handle binary masks folder as zip
        if file_type == 'binary_masks':
            masks_dir = os.path.join(output_dir, 'binary_masks')
            if not os.path.exists(masks_dir):
                return jsonify({'error': 'Binary masks not found'}), 404

            zip_path = os.path.join(output_dir, 'binary_masks.zip')
            if not os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file in os.listdir(masks_dir):
                        file_path = os.path.join(masks_dir, file)
                        zipf.write(file_path, file)

            return send_file(
                zip_path,
                as_attachment=True,
                download_name=f'{job_id}_binary_masks.zip'
            )

        # Handle vectors folder as zip
        if file_type == 'vectors':
            vectors_dir = os.path.join(output_dir, 'vectors')
            if not os.path.exists(vectors_dir):
                return jsonify({'error': 'Vectors not found'}), 404

            zip_path = os.path.join(output_dir, 'vectors.zip')
            if not os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file in os.listdir(vectors_dir):
                        file_path = os.path.join(vectors_dir, file)
                        zipf.write(file_path, file)

            return send_file(
                zip_path,
                as_attachment=True,
                download_name=f'{job_id}_vectors.zip'
            )

        # Map file types to directories
        type_to_dir = {
            'overlay': 'debug_overlays',
            'mask': 'masks',
            'filtered': 'filtered_images',
            'filter_mask': 'filtered_images',
            'report': '' 
        }

        subdir = type_to_dir.get(file_type, '')

        if filename:
            if subdir:
                file_path = os.path.join(output_dir, subdir, filename)
            else:
                file_path = os.path.join(output_dir, filename)

            if os.path.exists(file_path):
                return send_file(file_path, as_attachment=True)

        return jsonify({'error': 'File not found'}), 404

    @bp.route('/api/download/<job_id>/processing_stats', methods=['GET'])
    def download_processing_stats(job_id):
        output_dir = os.path.join(config.OUTPUT_FOLDER, job_id)
        stats_path = os.path.join(output_dir, 'processing_stats.json')

        if os.path.exists(stats_path):
            return send_file(stats_path, as_attachment=True)

        return jsonify({'error': 'Processing stats not found'}), 404

    @bp.route('/api/download/<job_id>/result_json', methods=['GET'])
    def download_result_json(job_id):
        output_dir = os.path.join(config.OUTPUT_FOLDER, job_id)
        result_path = os.path.join(output_dir, 'result.json')

        if os.path.exists(result_path):
            return send_file(result_path, as_attachment=True)

        return jsonify({'error': 'Result JSON not found'}), 404

    return bp
