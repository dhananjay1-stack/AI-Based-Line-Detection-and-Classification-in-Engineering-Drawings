import os
import sys
import uuid
import json
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file, send_from_directory, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import threading
import time

sys.path.insert(0, str(Path(__file__).parent))

from inference_engine import InferenceEngine
from config import Config

app = Flask(__name__,
            static_folder='static',
            template_folder='templates')
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = Config.OUTPUT_FOLDER

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)

# Global inference engine
inference_engine = None

# Job storage
jobs = {}
jobs_lock = threading.Lock()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_inference_engine():
    global inference_engine
    if inference_engine is None:
        print("Initializing InferenceEngine...")
        try:
            inference_engine = InferenceEngine(
                line_detection_model_path=Config.DEFAULT_CHECKPOINT,
                legend_path=Config.LEGEND_PATH,
                num_classes=Config.NUM_CLASSES,
                backbone=Config.BACKBONE,
                device=Config.DEVICE,
                tile_size=Config.TILE_SIZE,
                overlap=Config.OVERLAP,
                temperature=Config.TEMPERATURE,
            )
            print("InferenceEngine initialized successfully")
        except Exception as e:
            print(f"Failed to initialize InferenceEngine: {e}")
            import traceback
            traceback.print_exc()
            inference_engine = None
    return inference_engine


def process_job(job_id, files, params):
    """Background job processor — line detection with enhanced visualizations."""
    try:
        with jobs_lock:
            jobs[job_id]['status'] = 'processing'

        engine = get_inference_engine()

        # switch model if needed
        checkpoint = params.get('checkpoint', '')
        if checkpoint and checkpoint != engine._current_checkpoint:
            if not engine.switch_model(checkpoint):
                raise RuntimeError(f"Failed to load checkpoint: {checkpoint}")

        results = []
        for i, file_path in enumerate(files):
            with jobs_lock:
                jobs[job_id]['progress'] = {
                    'current': i + 1,
                    'total': len(files),
                    'filename': os.path.basename(file_path)
                }

            result = engine.process_image(
                image_path=file_path,
                job_id=job_id,
                output_dir=os.path.join(Config.OUTPUT_FOLDER, job_id),
                use_tta=params.get('use_tta', False),
                enhance_contrast=Config.ENHANCE_CONTRAST,
                classes_to_show=params.get('classes_to_show'),
                confidence_threshold=params.get('confidence', 0.0),
                dilate=params.get('dilate', 0),
                include_visualizations=params.get('include_visualizations', True),
            )
            results.append(result)

        with jobs_lock:
            jobs[job_id]['status'] = 'done'
            jobs[job_id]['results'] = results
            jobs[job_id]['completed_at'] = datetime.now().isoformat()

    except Exception as e:
        import traceback
        traceback.print_exc()
        with jobs_lock:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['error'] = str(e)


# ── Routes ───────────────────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/checkpoints', methods=['GET'])
def get_checkpoints():
    """List available model checkpoints."""
    ckpts = Config.list_checkpoints()
    default = Config.DEFAULT_CHECKPOINT
    return jsonify({'checkpoints': ckpts, 'default': default})


@app.route('/api/classes', methods=['GET'])
def get_classes():
    """List all line-type classes with colours."""
    return jsonify({'classes': Config.get_class_info()})


@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Upload one or more files (or a zip)."""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400

    job_id = str(uuid.uuid4())
    job_dir = os.path.join(Config.UPLOAD_FOLDER, job_id)
    os.makedirs(job_dir, exist_ok=True)

    uploaded_files = []
    for file in files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            file_path = os.path.join(job_dir, filename)

            if filename.lower().endswith('.zip'):
                file.save(file_path)
                with zipfile.ZipFile(file_path, 'r') as zf:
                    zf.extractall(job_dir)
                os.remove(file_path)
                for root, _, extracted in os.walk(job_dir):
                    for ef in extracted:
                        if allowed_file(ef):
                            uploaded_files.append(os.path.join(root, ef))
            elif allowed_file(filename):
                file.save(file_path)
                uploaded_files.append(file_path)

    if not uploaded_files:
        shutil.rmtree(job_dir, ignore_errors=True)
        return jsonify({'error': 'No valid image files found'}), 400

    with jobs_lock:
        jobs[job_id] = {
            'job_id': job_id,
            'status': 'uploaded',
            'files': uploaded_files,
            'file_count': len(uploaded_files),
            'created_at': datetime.now().isoformat(),
            'filenames': [os.path.basename(f) for f in uploaded_files]
        }

    return jsonify({
        'job_id': job_id,
        'file_count': len(uploaded_files),
        'filenames': [os.path.basename(f) for f in uploaded_files]
    })


@app.route('/api/run', methods=['POST'])
def run_processing():
    """Start line-detection job with custom parameters."""
    data = request.get_json()
    if not data or 'job_id' not in data:
        return jsonify({'error': 'job_id required'}), 400

    job_id = data['job_id']

    with jobs_lock:
        if job_id not in jobs:
            return jsonify({'error': 'Job not found'}), 404
        job = jobs[job_id]
        if job['status'] not in ['uploaded', 'done', 'failed']:
            return jsonify({'error': 'Job already processing'}), 400
        jobs[job_id]['status'] = 'queued'

    output_dir = os.path.join(Config.OUTPUT_FOLDER, job_id)
    os.makedirs(output_dir, exist_ok=True)

    # Collect parameters from request
    params = {
        'checkpoint': data.get('checkpoint', ''),
        'classes_to_show': data.get('classes'),         # list[int] or None
        'use_tta': bool(data.get('use_tta', False)),
        'confidence': float(data.get('confidence', 0.0)),
        'dilate': int(data.get('dilate', 0)),
        'include_visualizations': bool(data.get('include_visualizations', True)),
    }

    thread = threading.Thread(
        target=process_job,
        args=(job_id, job['files'], params)
    )
    thread.daemon = True
    thread.start()

    return jsonify({'job_id': job_id, 'status': 'queued', 'params': params})


@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    with jobs_lock:
        if job_id not in jobs:
            return jsonify({'error': 'Job not found'}), 404
        job = jobs[job_id].copy()
    if 'files' in job:
        del job['files']
    return jsonify(job)


@app.route('/api/result/<job_id>', methods=['GET'])
def get_result(job_id):
    with jobs_lock:
        if job_id not in jobs:
            return jsonify({'error': 'Job not found'}), 404
        job = jobs[job_id].copy()

    if job['status'] != 'done':
        return jsonify({'job_id': job_id, 'status': job['status'], 'error': job.get('error')})

    return jsonify({
        'job_id': job_id,
        'status': 'done',
        'results': job.get('results', [])
    })


@app.route('/api/files/<job_id>/<path:filename>', methods=['GET'])
def serve_file(job_id, filename):
    file_path = os.path.join(Config.OUTPUT_FOLDER, job_id, filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    return jsonify({'error': 'File not found'}), 404


@app.route('/api/download/<job_id>/<file_type>', methods=['GET'])
def download_file(job_id, file_type):
    output_dir = os.path.join(Config.OUTPUT_FOLDER, job_id)
    if not os.path.exists(output_dir):
        return jsonify({'error': 'Output not found'}), 404

    if file_type == 'all_zip':
        zip_path = os.path.join(output_dir, 'all_outputs.zip')
        if not os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        if file != 'all_outputs.zip':
                            fp = os.path.join(root, file)
                            zipf.write(fp, os.path.relpath(fp, output_dir))
        return send_file(zip_path, as_attachment=True, download_name=f'{job_id}_outputs.zip')

    filename = request.args.get('filename', '')
    if filename:
        file_path = os.path.join(output_dir, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)

    return jsonify({'error': 'File not found'}), 404


@app.route('/api/history', methods=['GET'])
def get_history():
    with jobs_lock:
        history = []
        for jid, job in jobs.items():
            history.append({
                'job_id': jid,
                'status': job['status'],
                'file_count': job.get('file_count', 0),
                'created_at': job.get('created_at'),
                'completed_at': job.get('completed_at')
            })
    history.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return jsonify(history[:20])


if __name__ == '__main__':
    print("=" * 60)
    print("  Engineering Drawing — Line Detection Server")
    print("=" * 60)
    print(f"  Default Checkpoint : {Config.DEFAULT_CHECKPOINT}")
    print(f"  Legend             : {Config.LEGEND_PATH}")
    print(f"  Device             : {Config.DEVICE}")
    print(f"  Temperature        : {Config.TEMPERATURE}")
    print(f"  Tile Size          : {Config.TILE_SIZE}")
    print(f"  Overlap            : {Config.OVERLAP}")
    ckpts = Config.list_checkpoints()
    print(f"  Checkpoints found  : {len(ckpts)}")
    for c in ckpts:
        print(f"    - {c['name']}  ({c['size_mb']} MB)")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
