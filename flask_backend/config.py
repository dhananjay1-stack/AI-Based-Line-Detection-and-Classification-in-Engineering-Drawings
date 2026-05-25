import os
import torch
from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).parent.parent.parent
    APPROACH2_DIR = Path(__file__).parent.parent
    FLASK_DIR = Path(__file__).parent

    # ── Model paths ──────────────────────────────────────────────────
    # Directory scanned for .pth checkpoints (user picks from UI)
    CHECKPOINTS_DIR = str(APPROACH2_DIR / "finetune_annotation_output" / "checkpoints")
    DEFAULT_CHECKPOINT = str(APPROACH2_DIR / "finetune_annotation_output" / "checkpoints" / "best_model.pth")
    LEGEND_PATH = str(APPROACH2_DIR / "legend.json")

    NUM_CLASSES = 11
    BACKBONE = "resnet50"

    # Inference defaults (match inference.py)
    TILE_SIZE = 512
    OVERLAP = 128
    TEMPERATURE = 0.5       # lower = sharper predictions

    # Post-processing defaults
    ALPHA_EDGE_BOOST = 2.0
    DEFAULT_THRESHOLD = 0.08
    MIN_CONFIDENCE = 0.02
    FORCE_ARGMAX = True
    USE_BILATERAL = True
    USE_CRF = False
    ENHANCE_CONTRAST = True

    # UI-controlled defaults (can be overridden per job)
    USE_TTA = False
    DEFAULT_CONFIDENCE_FILTER = 0.0
    DEFAULT_DILATE = 0

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    UPLOAD_FOLDER = str(FLASK_DIR / "uploads")
    OUTPUT_FOLDER = str(FLASK_DIR / "outputs")

    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB

    # Excluded classes (matched with inference.py)
    EXCLUDED_CLASSES = {6, 7, 9}

    # Class definitions (exact match with inference.py CLASS_COLORS fallback)
    CLASS_NAMES = [
        "Background",
        "Center_line",
        "Dimension_lines",
        "Extension_line",
        "Feature_Visible",
        "Leader_line",
        "Phantom_line",
        "break_line",
        "cutting_plane",
        "hidden_line",
        "Section_hatching",
    ]

    # BGR colors for visualization (exactly matching inference.py CLASS_COLORS)
    CLASS_COLORS_RGB = {
        0:  [40, 40, 40],           # Background
        1:  [255, 60, 60],          # Center_line (Red)
        2:  [60, 220, 60],          # Dimension_lines (Green)
        3:  [60, 100, 255],         # Extension_line (Blue)
        4:  [255, 230, 50],         # Feature_Visible (Yellow)
        5:  [230, 60, 230],         # Leader_line (Magenta)
        8:  [60, 200, 160],         # cutting_plane (Teal)
        10: [50, 200, 120],         # Section_hatching (Emerald)
    }

    # Hex colors for UI (from RGB)
    CLASS_COLORS_HEX = {
        0:  "#282828",
        1:  "#FF3C3C",
        2:  "#3CDC3C",
        3:  "#3C64FF",
        4:  "#FFE632",
        5:  "#E63CE6",
        8:  "#3CC8A0",
        10: "#32C878",
    }

    MAX_JOB_AGE_HOURS = 24
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = True

    @classmethod
    def ensure_dirs(cls):
        """Ensure all required directories exist."""
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(cls.OUTPUT_FOLDER, exist_ok=True)
        os.makedirs(cls.CHECKPOINTS_DIR, exist_ok=True)

    @classmethod
    def list_checkpoints(cls):
        """Scan CHECKPOINTS_DIR for .pth / .pt checkpoint files."""
        ckpts = []
        ckpt_dir = Path(cls.CHECKPOINTS_DIR)
        if ckpt_dir.exists():
            for f in sorted(ckpt_dir.glob("*.pth")):
                ckpts.append({'name': f.stem, 'path': str(f), 'size_mb': round(f.stat().st_size / 1e6, 1)})
            for f in sorted(ckpt_dir.glob("*.pt")):
                if not any(c['path'] == str(f) for c in ckpts):
                    ckpts.append({'name': f.stem, 'path': str(f), 'size_mb': round(f.stat().st_size / 1e6, 1)})
        return ckpts

    @classmethod
    def get_class_info(cls):
        """Return non-background class list with colors for frontend."""
        classes = []
        for idx, name in enumerate(cls.CLASS_NAMES):
            if idx == 0 or idx in cls.EXCLUDED_CLASSES:
                continue
            color_hex = cls.CLASS_COLORS_HEX.get(idx, '#808080')
            color_rgb = cls.CLASS_COLORS_RGB.get(idx, [128, 128, 128])
            classes.append({
                'index': idx,
                'name': name,
                'color': color_hex,
                'color_rgb': color_rgb,
            })
        return classes

    @classmethod
    def get_model_info(cls):
        """Get model paths and availability info."""
        return {
            'line_detection': {
                'path': cls.DEFAULT_CHECKPOINT,
                'exists': os.path.exists(cls.DEFAULT_CHECKPOINT)
            },
            'legend': {
                'path': cls.LEGEND_PATH,
                'exists': os.path.exists(cls.LEGEND_PATH)
            },
            'device': cls.DEVICE,
            'checkpoints': cls.list_checkpoints(),
            'settings': {
                'tile_size': cls.TILE_SIZE,
                'overlap': cls.OVERLAP,
                'temperature': cls.TEMPERATURE,
                'use_tta': cls.USE_TTA,
                'force_argmax': cls.FORCE_ARGMAX,
                'default_threshold': cls.DEFAULT_THRESHOLD,
                'min_confidence': cls.MIN_CONFIDENCE,
                'use_bilateral': cls.USE_BILATERAL,
                'enhance_contrast': cls.ENHANCE_CONTRAST
            }
        }
