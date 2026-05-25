# Engineering Drawing Processor

A Flask-based web application for processing engineering drawings with line detection and non-essential element filtering capabilities.

## Features

- **Detection Mode**: Detect and highlight different line types in engineering drawings
- **Filter Mode**: Remove non-essential parts using YOLO segmentation
- **Combined Mode**: Filter first, then detect lines on cleaned image
- **Batch Processing**: Upload multiple images or ZIP files
- **Interactive Results**: View overlays, compare original vs processed images
- **Detailed Reports**: Confidence scores, legends, and downloadable outputs

## Line Types Detected

| Class | Color |
|-------|-------|
| Center Line | Red |
| Dimension Line | Green |
| Extension Line | Blue |
| Feature Visible | Yellow |
| Leader Line | Magenta |
| Phantom Line | Cyan |
| Section Hatching | Orange |
| Break Line | Purple |
| Cutting Plane | Teal |
| Hidden Line | Light Pink |

## Project Structure (Modular Architecture)

```
flask_backend/
├── app.py                      # Main Flask application entry point
├── config.py                   # Configuration settings
├── inference_engine.py         # Legacy inference engine (deprecated)
├── requirements.txt            # Python dependencies
│
├── routes/                     # API route handlers
│   ├── __init__.py
│   ├── upload.py               # File upload endpoints
│   ├── process.py              # Processing endpoints
│   ├── result.py               # Result retrieval endpoints
│   └── download.py             # Download endpoints
│
├── services/                   # Business logic layer
│   ├── __init__.py
│   ├── job_service.py          # Job lifecycle management
│   ├── line_detection_service.py   # Line detection orchestration
│   ├── filter_out_service.py   # Filter-out orchestration
│   └── result_service.py       # Result formatting and storage
│
├── inference/                  # ML inference modules
│   ├── __init__.py
│   ├── line_detection_inference.py   # Line detection model inference
│   └── filter_out_inference.py       # Filter-out model inference
│
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── file_utils.py           # File operations
│   ├── image_utils.py          # Image processing utilities
│   ├── json_utils.py           # JSON helpers
│   ├── logging_utils.py        # Logging utilities
│   └── path_utils.py           # Path management
│
├── static/
│   ├── css/styles.css          # Frontend styles
│   └── js/app.js               # Frontend JavaScript
│
├── templates/
│   └── index.html              # Main HTML template
│
├── test_line_detection.py      # Line detection test script
├── test_filter_out.py          # Filter-out test script
├── test_combined.py            # Combined pipeline test script
│
├── sample_result.json          # Example result JSON
├── sample_processing_stats.json    # Example processing stats
│
├── uploads/                    # Uploaded files (created at runtime)
└── outputs/                    # Processing outputs (created at runtime)
```

## Output Structure

For each job, outputs are organized as follows:

```
outputs/<job_id>/
├── input/                      # Copies of input images
├── masks/                      # Argmax segmentation masks
├── binary_masks/               # Per-class binary masks
│   ├── <basename>_center_line.png
│   ├── <basename>_dimension_line.png
│   └── ...
├── debug_overlays/             # Colored overlay visualizations
│   ├── <basename>_overlay.png
│   └── <basename>_filter_overlay.png
├── filtered_images/            # Filtered images (filter mode)
│   ├── <basename>_filtered.png
│   └── <basename>_filter_mask.png
├── vectors/                    # Skeleton and segment data
│   ├── <basename>_skeleton.png
│   └── <basename>_segments.json
├── processing_stats.json       # Processing statistics
├── result.json                 # Full results
└── <basename>_report.json      # Per-image reports
```

## Setup

### 1. Install Dependencies

```bash
cd flask_backend
pip install -r requirements.txt
```

### 2. Verify Model Paths

The application expects models at these locations (configured in `config.py`):

- **Line Detection Model**: `checkpoints/best.pth`
- **Filter Model (YOLO)**: Path to YOLO segmentation model

Update `config.py` if your model paths differ.

### 3. Run the Application

```bash
python app.py
```

The server will start at `http://localhost:5000`

## Usage

1. **Open Browser**: Navigate to `http://localhost:5000`

2. **Upload Images**:
   - Drag and drop images onto the upload area
   - Or click to browse and select files
   - Supports: PNG, JPG, JPEG, TIFF, BMP, ZIP

3. **Select Processing Mode**:
   - **Detection Only**: Detect and color-code line types
   - **Filter Only**: Remove non-essential elements
   - **Combined**: Filter then detect (both)

4. **Run Processing**: Click "Run Processing" and wait for results

5. **View Results**:
   - Switch between overlay, original, filtered views
   - Use comparison mode with opacity slider
   - View legend with class colors and counts
   - Check confidence reports

6. **Download Outputs**:
   - Detection overlay image
   - Filtered image
   - Segmentation mask
   - Binary masks (ZIP)
   - Vectors (ZIP)
   - JSON reports
   - All outputs as ZIP

## API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main web interface |
| GET | `/api/health` | Health check with model status |
| GET | `/api/models` | Model information |

### Upload & Process

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload files (multipart/form-data) |
| POST | `/api/run` | Start processing (JSON: `{job_id, mode}`) |
| GET | `/api/status/{job_id}` | Get job status and progress |

### Results

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/result/{job_id}` | Get full job results |
| GET | `/api/result/{job_id}/{filename}` | Get single file result |
| GET | `/api/files/{job_id}/{path}` | Serve output files |
| GET | `/api/history` | Get recent job history |

### Downloads

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/download/{job_id}/all_zip` | Download all outputs as ZIP |
| GET | `/api/download/{job_id}/binary_masks` | Download binary masks as ZIP |
| GET | `/api/download/{job_id}/vectors` | Download vectors as ZIP |
| GET | `/api/download/{job_id}/overlay?filename=...` | Download specific overlay |
| GET | `/api/download/{job_id}/mask?filename=...` | Download specific mask |
| GET | `/api/download/{job_id}/processing_stats` | Download processing_stats.json |

### Job Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| DELETE | `/api/job/{job_id}` | Delete job and its files |

## Test Scripts

### Test Line Detection

```bash
python test_line_detection.py --image path/to/drawing.png --output test_output/
```

Options:
- `--image`: Input image path (required)
- `--output`: Output directory (default: test_output)
- `--tta`: Enable test-time augmentation

### Test Filter-Out

```bash
python test_filter_out.py --image path/to/drawing.png --output test_filter_output/
```

Options:
- `--image`: Input image path (required)
- `--output`: Output directory (default: test_filter_output)
- `--threshold`: Confidence threshold override

### Test Combined Pipeline

```bash
python test_combined.py --image path/to/drawing.png --output test_combined_output/
```

Options:
- `--image`: Input image path (required)
- `--output`: Output directory (default: test_combined_output)
- `--tta`: Enable test-time augmentation

## Configuration

Edit `config.py` to customize:

```python
class Config:
    # Model paths
    LINE_DETECTION_MODEL_PATH = "path/to/line_model.pth"
    FILTER_MODEL_PATH = "path/to/filter_model.pt"

    # Model configuration
    NUM_CLASSES = 11              # Number of line classes + background
    BACKBONE = "resnet50"         # Model backbone

    # Inference settings
    TILE_SIZE = 1024              # Sliding window tile size
    OVERLAP = 256                 # Tile overlap
    USE_TTA = False               # Test-time augmentation

    # Filter settings
    FILTER_CONF_THRESHOLD = 0.5   # YOLO confidence threshold
    MAX_YOLO_IMGSZ = 2048         # Max image size for YOLO

    # Device
    DEVICE = "cuda"               # or "cpu"

    # Server settings
    HOST = "0.0.0.0"
    PORT = 5000
    DEBUG = True
```

## Result JSON Format

```json
{
  "job_id": "abc12345-...",
  "mode": "both",
  "status": "completed",
  "results": [
    {
      "filename": "drawing1.png",
      "line_detection": {
        "overlay_path": "/api/files/.../overlay.png",
        "confidence_score": 0.91,
        "class_counts": {"center_line": 12450, ...},
        "class_confidences": {"center_line": 0.89, ...}
      },
      "filter_out": {
        "filtered_image_path": "/api/files/.../filtered.png",
        "confidence_score": 0.94,
        "removed_count": 145230,
        "retained_count": 4854770
      },
      "processing_time": 3.84,
      "warnings": []
    }
  ]
}
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- Flask 2.3+
- flask-cors
- ultralytics (for YOLO filter model)
- segmentation-models-pytorch
- scikit-image
- opencv-python
- Pillow
- numpy
- tqdm

## Troubleshooting

### CUDA Out of Memory
- Reduce `TILE_SIZE` in config.py
- Set `DEVICE = "cpu"` for CPU-only processing
- Process smaller batches

### Model Not Found
- Verify model paths in `config.py`
- Ensure models are trained and saved correctly
- Check `/api/health` endpoint for model status

### Slow Processing
- Enable CUDA if GPU available
- Reduce image size before upload
- Disable TTA (test-time augmentation)
- Increase `OVERLAP` to reduce tile count

### Import Errors
- Ensure all dependencies are installed
- Check Python path includes parent directories
- Run from flask_backend directory

## Architecture Overview

```
[Frontend] <-> [Routes] <-> [Services] <-> [Inference]
                  |              |              |
            Flask Blueprints  Job Management   ML Models
                  |              |              |
            upload.py       job_service.py   line_detection_inference.py
            process.py      result_service.py  filter_out_inference.py
            result.py       ...
            download.py
```

- **Routes**: Thin HTTP handlers, validation, response formatting
- **Services**: Business logic, orchestration, job management
- **Inference**: ML model loading and inference
- **Utils**: Shared utilities for files, images, JSON, logging, paths
