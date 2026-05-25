import os
from flask import Blueprint, jsonify, send_file


def create_result_routes(job_service, config):
    bp = Blueprint('result', __name__)

    @bp.route('/api/result/<job_id>', methods=['GET'])
    def get_result(job_id):
        job = job_service.get_job(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404

        if job['status'] != job_service.STATUS_COMPLETED:
            return jsonify({
                'job_id': job_id,
                'status': job['status'],
                'error': job.get('error')
            })

        response = {
            'job_id': job_id,
            'status': 'done',
            'mode': job.get('mode', 'detection'),
            'results': job.get('results', []),
            'warnings': job.get('warnings', [])
        }

        return jsonify(response)

    @bp.route('/api/result/<job_id>/<filename>', methods=['GET'])
    def get_single_result(job_id, filename):
        job = job_service.get_job(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404

        if job['status'] != job_service.STATUS_COMPLETED:
            return jsonify({
                'job_id': job_id,
                'status': job['status'],
                'error': job.get('error')
            })

        results = job.get('results', [])
        for result in results:
            if result.get('filename') == filename:
                return jsonify({
                    'job_id': job_id,
                    'status': 'done',
                    'result': result
                })

        return jsonify({'error': 'File not found in results'}), 404

    @bp.route('/api/files/<job_id>/<path:filename>', methods=['GET'])
    def serve_file(job_id, filename):
        file_path = os.path.join(config.OUTPUT_FOLDER, job_id, filename)

        if os.path.exists(file_path):
            return send_file(file_path)

        upload_path = os.path.join(config.UPLOAD_FOLDER, job_id, filename)
        if os.path.exists(upload_path):
            return send_file(upload_path)

        return jsonify({'error': 'File not found'}), 404

    @bp.route('/api/history', methods=['GET'])
    def get_history():
        history = job_service.get_history(limit=20)
        return jsonify(history)

    @bp.route('/api/job/<job_id>', methods=['DELETE'])
    def delete_job(job_id):
        if job_service.delete_job(job_id):
            return jsonify({'success': True, 'message': 'Job deleted'})

        return jsonify({'error': 'Job not found'}), 404

    return bp
