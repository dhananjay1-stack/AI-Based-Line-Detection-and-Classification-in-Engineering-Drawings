"""
Training Configuration for Thin-Line Segmentation Fine-Tuning
==============================================================

Central config dataclass. Edit values here to control the entire
training pipeline without touching any other file.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class TrainConfig:
    """All hyperparameters and paths for the fine-tuning pipeline."""

    # ── Paths ──────────────────────────────────────────────────────────
    # Cleaned patch dataset produced by patch_pipeline.py
    patch_root: str = r"C:\Users\Admin\line_detection\Approach 2\dataset_patches"
    # Class mapping JSON from dataset_pipeline.py
    classes_json: str = r"C:\Users\Admin\line_detection\Approach 2\pipeline_output\classes_cleaned.json"
    # Pretrained checkpoint to resume from
    checkpoint_path: str = r"C:\Users\Admin\line_detection\Approach 2\Segmentation_Deeplab_models_checkpoints\best_model.pth"
    # Output directory for new checkpoints, logs, previews
    out_dir: str = r"C:\Users\Admin\line_detection\Approach 2\finetune_output"

    # ── Model Architecture ─────────────────────────────────────────────
    backbone: str = "resnet50"          # Must match the checkpoint's encoder
    num_classes: int = 10               # 9 line classes + 1 background
    old_num_classes: int = 13           # Class count in the existing checkpoint
    in_channels: int = 3
    use_edge_head: bool = True          # Auxiliary centerline/edge head

    # ── Training Phases ────────────────────────────────────────────────
    # Phase 1: freeze encoder, train decoder + heads
    phase1_epochs: int = 15
    phase1_lr: float = 3e-4             # Higher LR for new heads

    # Phase 2: unfreeze encoder, train everything
    phase2_epochs: int = 35
    phase2_lr_backbone: float = 1e-5    # Very low for pretrained backbone
    phase2_lr_decoder: float = 5e-5     # Moderate for decoder (partially trained)
    phase2_lr_heads: float = 1e-4       # Higher for new heads

    # ── Optimizer ──────────────────────────────────────────────────────
    optimizer: str = "AdamW"
    weight_decay: float = 1e-4
    grad_clip_max_norm: float = 1.0
    warmup_epochs: int = 3              # Linear warmup before cosine
    scheduler: str = "cosine"           # "cosine" or "onecycle"

    # ── Loss Weights ───────────────────────────────────────────────────
    loss_ce_weight: float = 1.0
    loss_dice_weight: float = 1.0
    loss_focal_weight: float = 0.5      # Focal loss weight (0 to disable)
    loss_edge_weight: float = 0.5       # Lambda for edge/centerline loss
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25

    # ── Data Loading ───────────────────────────────────────────────────
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    use_weighted_sampler: bool = True    # Use sampling_weight from metadata
    patch_size: int = 512               # Input size (must match patch extraction)

    # ── Training Options ───────────────────────────────────────────────
    use_amp: bool = True                # Mixed precision (FP16)
    grad_accum_steps: int = 1           # Gradient accumulation (effective BS = batch_size * this)
    seed: int = 42
    early_stopping_patience: int = 12
    resume: bool = False                # Resume from last checkpoint

    # ── Logging ────────────────────────────────────────────────────────
    log_every_n_steps: int = 50
    save_preview_every_n_epochs: int = 5
    num_preview_samples: int = 8

    # ── Augmentation ───────────────────────────────────────────────────
    aug_hflip: bool = True
    aug_vflip: bool = True
    aug_rotate_limit: int = 15
    aug_brightness_contrast: bool = True
    aug_elastic: bool = False           # Can distort thin lines


@dataclass
class ClassInfo:
    """Resolved class information from the JSON mapping."""
    id_to_name: dict = field(default_factory=dict)
    name_to_id: dict = field(default_factory=dict)
    num_classes: int = 10

    @classmethod
    def from_json(cls, json_path: str) -> "ClassInfo":
        import json
        with open(json_path) as f:
            cmap = json.load(f)
        id_to_name = {0: "Background"}
        name_to_id = {"Background": 0}
        for name, cid in cmap.items():
            id_to_name[cid] = name
            name_to_id[name] = cid
        return cls(
            id_to_name=id_to_name,
            name_to_id=name_to_id,
            num_classes=len(id_to_name),
        )
