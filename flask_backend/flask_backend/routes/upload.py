import os
import sys
import zipfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from utils.file_utils import allowed_file, ensure_dir
from utils.path_utils import generate_job_id
from utils.logging_utils import log_upload


def create_upload_routes(job_service, config):
    bp = Blueprint('upload', __name__)

    @bp.route('/api/upload', methods=['POST'])
    def upload_files():
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400

        job_id = generate_job_id()
        job_dir = os.path.join(config.UPLOAD_FOLDER, job_id)
        input_dir = os.path.join(job_dir, 'input')
        ensure_dir(input_dir)

        uploaded_files = []
        filenames = []

        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(input_dir, filename)

                # Handle zip files
                if filename.lower().endswith('.zip'):
                    file.save(file_path)

                    try:
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            zip_ref.extractall(input_dir)
                        os.remove(file_path)

                        for root, dirs, extracted_files in os.walk(input_dir):
                            for ef in extracted_files:
                                if allowed_file(ef):
                                    full_path = os.path.join(root, ef)
                                    uploaded_files.append(full_path)
                                    filenames.append(os.path.basename(ef))
                    except zipfile.BadZipFile:
                        return jsonify({'error': 'Invalid zip file'}), 400

                elif allowed_file(filename):
                    file.save(file_path)
                    uploaded_files.append(file_path)
                    filenames.append(filename)

        if not uploaded_files:
            shutil.rmtree(job_dir, ignore_errors=True)
            return jsonify({'error': 'No valid image files found'}), 400

        job_service.create_job(job_id, uploaded_files, filenames)
        log_upload(job_id, len(uploaded_files), filenames)

        return jsonify({
            'job_id': job_id,
            'file_count': len(uploaded_files),
            'filenames': filenames
        })

    return bp
