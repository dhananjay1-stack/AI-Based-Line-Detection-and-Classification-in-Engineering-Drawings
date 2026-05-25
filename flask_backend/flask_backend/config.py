import os
import torch
from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).parent.parent.parent
    APPROACH2_DIR = Path(__file__).parent.parent
    FLASK_DIR = Path(__file__).parent

    # ── Model paths ──────────────────────────────────────────────────
    # Directory scanned for .pth checkpoints (user picks from UI)
    CHECKPOINTS_DIR = str(APPROACH2_DIR / "checkpoints")
    DEFAULT_CHECKPOINT = r"D:\Practice\AI\Emotion_Based_Movies_Recommonation\line_detection\Approach2\checkpoints\best.pth"
    LEGEND_PATH = r"D:\Practice\AI\Emotion_Based_Movies_Recommonation\line_detection\Approach2\legend.json"

    NUM_CLASSES = 11  # 10 line types + background
    BACKBONE = "resnet50"

    # Classes excluded from inference results, visualization, and reports
    # 6 = phantom_line, 7 = section_hatching (mapped to break_line in Approach2),
    # but using Approach2 index mapping: 6=phantom, 7=break, 9=hidden
    # NOTE: In this Flask config the index mapping is:
    #   6=phantom_line, 8=break_line, 10=hidden_line
    # We exclude by NAME to stay consistent across index schemes.
    EXCLUDED_CLASS_NAMES = {"phantom_line", "break_line", "hidden_line"}
    EXCLUDED_CLASS_INDICES = set()  # populated at class load time

    @classmethod
    def _init_excluded_indices(cls):
        """Populate EXCLUDED_CLASS_INDICES from EXCLUDED_CLASS_NAMES."""
        cls.EXCLUDED_CLASS_INDICES = {
            idx for idx, name in enumerate(cls.CLASS_NAMES)
            if name in cls.EXCLUDED_CLASS_NAMES
        }

    # ── Inference defaults ───────────────────────────────────────────
    TILE_SIZE = 1024
    OVERLAP = 256
    TEMPERATURE = 0.5       # lower = sharper predictions

    # ── Post-processing defaults ─────────────────────────────────────
    ALPHA_EDGE_BOOST = 2.0
    DEFAULT_THRESHOLD = 0.08
    MIN_CONFIDENCE = 0.02
    FORCE_ARGMAX = True
    USE_BILATERAL = True
    USE_CRF = False
    ENHANCE_CONTRAST = True

    # ── UI-controlled defaults (can be overridden per job) ───────────
    USE_TTA = False
    DEFAULT_CONFIDENCE_FILTER = 0.0   # softmax confidence gate
    DEFAULT_DILATE = 0                # px dilation for thin lines

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    UPLOAD_FOLDER = str(FLASK_DIR / "uploads")
    OUTPUT_FOLDER = str(FLASK_DIR / "outputs")

    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB

    # ── Class definitions ────────────────────────────────────────────
    CLASS_NAMES = [
        "background",
        "center_line",
        "dimension_line",
        "extension_line",
        "feature_visible",
        "leader_line",
        "phantom_line",
        "section_hatching",
        "break_line",
        "cutting_plane",
        "hidden_line",
    ]

    CLASS_COLORS = {
        "background":       "#000000",
        "center_line":      "#FF0000",
        "dimension_line":   "#00FF00",
        "extension_line":   "#0000FF",
        "feature_visible":  "#FFFF00",
        "leader_line":      "#FF00FF",
        "section_hatching": "#FFA500",
        "cutting_plane":    "#008080",
    }

    CLASS_COLORS_RGB = {
        "background":       [0, 0, 0],
        "center_line":      [255, 0, 0],
        "dimension_line":   [0, 255, 0],
        "extension_line":   [0, 0, 255],
        "feature_visible":  [255, 255, 0],
        "leader_line":      [255, 0, 255],
        "section_hatching": [255, 165, 0],
        "cutting_plane":    [0, 128, 128],
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
        """Return non-background, non-excluded class list with colors for frontend."""
        cls._init_excluded_indices()
        classes = []
        for idx, name in enumerate(cls.CLASS_NAMES):
            if idx == 0 or name in cls.EXCLUDED_CLASS_NAMES:
                continue
            classes.append({
                'index': idx,
                'name': name,
                'color': cls.CLASS_COLORS.get(name, '#808080'),
                'color_rgb': cls.CLASS_COLORS_RGB.get(name, [128, 128, 128]),
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
