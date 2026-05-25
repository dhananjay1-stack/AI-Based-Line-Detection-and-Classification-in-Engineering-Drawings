import os
import sys
import json
import csv
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    print("Install: pip install segmentation-models-pytorch albumentations")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("inference")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Defaults ──────────────────────────────────────────────────────────────

DEFAULT_CHECKPOINT = r"C:\Users\Admin\line_detection\Approach 2\finetune_output\checkpoints\best_model.pth"
FALLBACK_CHECKPOINT = r"C:\Users\Admin\line_detection\Approach 2\Segmentation_Deeplab_models_checkpoints\best_model.pth"
CLASSES_JSON = r"C:\Users\Admin\line_detection\Approach 2\pipeline_output\classes_cleaned.json"
DEFAULT_OUTPUT = r"C:\Users\Admin\line_detection\Approach 2\inference_output"

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}

# ── Class Colors ──────────────────────────────────────────────────────────

CLASS_COLORS = {
    0:  ("Background",      (40,  40,  40)),
    1:  ("Center_line",     (255, 60,  60)),      # Red
    2:  ("Dimension_lines", (60,  220, 60)),      # Green
    3:  ("Extension_line",  (60,  100, 255)),     # Blue
    4:  ("Feature_Visible", (255, 230, 50)),      # Yellow
    5:  ("Leader_line",     (230, 60,  230)),     # Magenta
    8:  ("cutting_plane",   (60,  200, 160)),     # Teal
    10: ("Section_hatching", (50,  200, 120)),     # Emerald
}

# Classes excluded from inference, visualization, and reports
# 6 = Phantom_lines, 7 = break_line, 9 = hidden_lines
EXCLUDED_CLASSES = {6, 7, 9}


# ── Model Loading ─────────────────────────────────────────────────────────

def load_model(checkpoint_path, num_classes=None):
    """
    Auto-detect checkpoint type and load model.
    Handles both old (13-class) and new (10-class) checkpoints,
    as well as the wrapped DeepLabV3PlusEdge and raw SMP models.
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    try:
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    # Determine state dict and metadata
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        epoch = ckpt.get("epoch", "?")
        miou = ckpt.get("best_miou", "?")
        phase = ckpt.get("phase", "?")
        logger.info(f"  Epoch: {epoch}, Phase: {phase}, mIoU: {miou}")
    else:
        state_dict = ckpt
        logger.info("  Raw state dict loaded")

    # Auto-detect num_classes from segmentation head
    detected_classes = None
    for key in state_dict:
        if "segmentation_head" in key and "weight" in key:
            detected_classes = state_dict[key].shape[0]
            break

    if num_classes is None:
        num_classes = detected_classes or 10
    logger.info(f"  Detected classes: {num_classes}")

    # Check if this is a wrapped model (has base_model prefix) or raw SMP
    has_base_model = any(k.startswith("base_model.") for k in state_dict)
    has_edge_head = any(k.startswith("edge_head.") for k in state_dict)

    if has_base_model or has_edge_head:
        # Load as our DeepLabV3PlusEdge wrapper
        sys.path.insert(0, str(Path(__file__).parent))
        from model import DeepLabV3PlusEdge
        model = DeepLabV3PlusEdge(
            num_classes=num_classes,
            backbone="resnet50",
            use_edge_head=has_edge_head,
        )
        model.load_state_dict(state_dict)
        logger.info(f"  Loaded as DeepLabV3PlusEdge (edge_head={'yes' if has_edge_head else 'no'})")
    else:
        # Load as raw SMP model
        model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
        )
        model.load_state_dict(state_dict)
        logger.info(f"  Loaded as raw SMP DeepLabV3Plus")

    model = model.to(DEVICE)
    model.eval()
    return model, num_classes


def load_class_names(json_path, num_classes):
    """Load class name mapping."""
    if os.path.exists(json_path):
        with open(json_path) as f:
            cmap = json.load(f)
        names = {0: "Background"}
        for name, cid in cmap.items():
            names[cid] = name
        return names
    else:
        return {i: CLASS_COLORS.get(i, (f"class_{i}", (128, 128, 128)))[0]
                for i in range(num_classes)}


# ── Inference Engine ──────────────────────────────────────────────────────

def get_transform():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_gaussian_window(size, sigma=0.5):
    x = np.linspace(-1, 1, size)
    g = np.exp(-0.5 * (x / sigma) ** 2)
    return np.outer(g, g).astype(np.float32)


def safe_imread(path):
    arr = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def run_inference(model, image_rgb, num_classes, transform, use_tta=True,
                  tile_size=512, overlap=128):
    """
    Tiled inference with Gaussian blending and optional TTA.
    Works for any image size.
    """
    h, w = image_rgb.shape[:2]
    stride = tile_size - overlap

    # For small images, just run directly
    if h <= tile_size and w <= tile_size:
        # Pad to tile_size
        padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        padded[:h, :w] = image_rgb
        tens = transform(image=padded)["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(tens)
            logits = out[0] if isinstance(out, tuple) else out
            if use_tta:
                out_flip = model(torch.flip(tens, [3]))
                logits_flip = out_flip[0] if isinstance(out_flip, tuple) else out_flip
                logits = (logits + torch.flip(logits_flip, [3])) / 2.0
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        return probs[:, :h, :w]

    # Tiled inference for large images
    window = get_gaussian_window(tile_size)
    prob_map = np.zeros((num_classes, h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)
            y1 = max(0, y2 - tile_size)
            x1 = max(0, x2 - tile_size)

            tile = image_rgb[y1:y2, x1:x2]
            th, tw = tile.shape[:2]

            if th < tile_size or tw < tile_size:
                pad_tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                pad_tile[:th, :tw] = tile
                tile = pad_tile

            tens = transform(image=tile)["image"].unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                out = model(tens)
                logits = out[0] if isinstance(out, tuple) else out
                if use_tta:
                    out_flip = model(torch.flip(tens, [3]))
                    logits_flip = out_flip[0] if isinstance(out_flip, tuple) else out_flip
                    logits = (logits + torch.flip(logits_flip, [3])) / 2.0
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]

            tile_h = min(tile_size, y2 - y1)
            tile_w = min(tile_size, x2 - x1)
            probs = probs[:, :tile_h, :tile_w]
            win = window[:tile_h, :tile_w]

            prob_map[:, y1:y2, x1:x2] += probs * win
            count_map[y1:y2, x1:x2] += win

    prob_map /= (count_map[np.newaxis, :, :] + 1e-6)
    return prob_map


# ── Visualization ─────────────────────────────────────────────────────────

def create_colored_overlay(image_rgb, pred_mask, alpha=0.45):
    """Create overlay with all classes shown in their assigned color (excluded classes omitted)."""
    colored = np.zeros_like(image_rgb)
    for cls_id, (_, color) in CLASS_COLORS.items():
        if cls_id == 0 or cls_id in EXCLUDED_CLASSES:
            continue
        colored[pred_mask == cls_id] = color

    overlay = cv2.addWeighted(image_rgb, 1.0 - alpha, colored, alpha, 0)
    return overlay


def create_per_class_masks(image_rgb, pred_mask, class_names, num_classes):
    """Create an individual highlighted view for each class (excluded classes omitted)."""
    masks = {}
    for cls_id in range(1, num_classes):
        if cls_id in EXCLUDED_CLASSES:
            continue

        cls_mask = (pred_mask == cls_id).astype(np.uint8)
        pixel_count = cls_mask.sum()

        if pixel_count == 0:
            continue

        name = class_names.get(cls_id, f"class_{cls_id}")
        color = CLASS_COLORS.get(cls_id, (name, (200, 200, 200)))[1]

        # Create: dimmed original + bright colored class pixels
        view = (image_rgb * 0.3).astype(np.uint8)
        mask_3ch = np.stack([cls_mask] * 3, axis=-1)
        colored_pixels = np.array(color, dtype=np.uint8)
        view = np.where(mask_3ch > 0, colored_pixels, view)

        # Add label text
        label = f"{name} (ID:{cls_id}, {pixel_count:,}px)"
        cv2.putText(view, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                     0.7, (255, 255, 255), 2, cv2.LINE_AA)

        masks[cls_id] = {"name": name, "view": view, "pixels": pixel_count}

    return masks


def create_confidence_heatmap(prob_map, image_rgb):
    """Create a heatmap showing prediction confidence."""
    confidence = np.max(prob_map, axis=0)  # (H, W)

    # Normalize and colorize
    conf_uint8 = (confidence * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(conf_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend with original
    blended = cv2.addWeighted(image_rgb, 0.4, heatmap_rgb, 0.6, 0)

    # Add colorbar text
    cv2.putText(blended, "Low Conf", (15, blended.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(blended, "High Conf", (blended.shape[1] - 130, blended.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return blended, confidence


def create_legend(class_names, num_classes):
    """Create a standalone color legend image (excluded classes omitted)."""
    # Build list of active class IDs for sizing
    active_ids = [c for c in range(num_classes) if c not in EXCLUDED_CLASSES]
    row_h = 35
    legend_h = len(active_ids) * row_h + 20
    legend = np.ones((legend_h, 300, 3), dtype=np.uint8) * 30

    cv2.putText(legend, "CLASS LEGEND", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    for idx, cls_id in enumerate(active_ids):
        y = (idx + 1) * row_h + 5
        name = class_names.get(cls_id, f"class_{cls_id}")
        color = CLASS_COLORS.get(cls_id, (name, (128, 128, 128)))[1]

        cv2.rectangle(legend, (10, y - 15), (30, y + 5), color, -1)
        cv2.rectangle(legend, (10, y - 15), (30, y + 5), (200, 200, 200), 1)
        cv2.putText(legend, f"{cls_id}: {name}", (40, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    return legend


def create_summary_panel(image_rgb, overlay, confidence_map, pred_mask,
                         class_names, num_classes):
    """Create a 2x2 panel: original | overlay | confidence | class distribution."""
    h, w = image_rgb.shape[:2]

    # Resize all to same size
    panel_h, panel_w = min(h, 512), min(w, 512)

    orig_resized = cv2.resize(image_rgb, (panel_w, panel_h))
    overlay_resized = cv2.resize(overlay, (panel_w, panel_h))
    conf_resized = cv2.resize(confidence_map, (panel_w, panel_h))

    # Class distribution chart (excluded classes omitted)
    chart = np.ones((panel_h, panel_w, 3), dtype=np.uint8) * 30
    unique, counts = np.unique(pred_mask, return_counts=True)
    total_fg = sum(c for u, c in zip(unique, counts) if u > 0 and u not in EXCLUDED_CLASSES)

    active_ids = [c for c in range(num_classes) if c not in EXCLUDED_CLASSES]
    num_active = len(active_ids)

    y_start = 30
    cv2.putText(chart, "Class Distribution", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    max_bar_w = panel_w - 120
    total_pixels = pred_mask.size

    for idx, cls_id in enumerate(active_ids):
        y = y_start + idx * (panel_h - 40) // num_active
        name = class_names.get(cls_id, f"c{cls_id}")[:12]
        color = CLASS_COLORS.get(cls_id, (name, (128, 128, 128)))[1]

        px_count = int((pred_mask == cls_id).sum())
        ratio = px_count / total_pixels
        bar_w = max(1, int(ratio * max_bar_w * 30))  # Scale up for visibility
        bar_w = min(bar_w, max_bar_w)

        cv2.rectangle(chart, (100, y - 5), (100 + bar_w, y + 12), color, -1)
        cv2.putText(chart, f"{name}", (5, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cv2.putText(chart, f"{ratio*100:.1f}%", (105 + bar_w, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

    # Labels on panels
    for img, label in [(orig_resized, "Original"),
                        (overlay_resized, "Prediction Overlay"),
                        (conf_resized, "Confidence Map"),
                        (chart, "Class Distribution")]:
        cv2.putText(img, label, (5, img.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Assemble 2x2
    top = np.hstack([orig_resized, overlay_resized])
    bottom = np.hstack([conf_resized, chart])
    panel = np.vstack([top, bottom])

    return panel


# ── Main Pipeline ─────────────────────────────────────────────────────────

def process_single_image(model, img_path, output_dir, class_names, num_classes,
                          transform, use_tta=True):
    """Process one image and save all outputs."""
    stem = Path(img_path).stem

    # Create per-image output directory
    img_out_dir = Path(output_dir) / stem
    img_out_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    img_bgr = safe_imread(img_path)
    if img_bgr is None:
        logger.warning(f"  Failed to read: {img_path}")
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # Run inference
    prob_map = run_inference(model, img_rgb, num_classes, transform, use_tta=use_tta)

    # Ignore phantom (6), break (7), and hidden lines (9)
    for c in [6, 7, 9]:
        if c < prob_map.shape[0]:
            prob_map[c, :, :] = 0.0

    # Predictions
    pred_mask = np.argmax(prob_map, axis=0).astype(np.uint8)

    # ── 1. Save original ──
    cv2.imwrite(str(img_out_dir / "1_original.jpg"), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # ── 2. Save colored overlay ──
    overlay = create_colored_overlay(img_rgb, pred_mask)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(img_out_dir / "2_prediction_overlay.jpg"), overlay_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 95])

    # ── 3. Save per-class masks ──
    class_masks = create_per_class_masks(img_rgb, pred_mask, class_names, num_classes)
    for cls_id, data in class_masks.items():
        view_bgr = cv2.cvtColor(data["view"], cv2.COLOR_RGB2BGR)
        cls_name = data["name"].replace(" ", "_")
        cv2.imwrite(str(img_out_dir / f"3_class_{cls_id:02d}_{cls_name}.jpg"),
                     view_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # ── 4. Save confidence heatmap ──
    conf_vis, confidence = create_confidence_heatmap(prob_map, img_rgb)
    conf_bgr = cv2.cvtColor(conf_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(img_out_dir / "4_confidence_map.jpg"), conf_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 90])

    # ── 5. Save raw mask (for further processing) ──
    Image.fromarray(pred_mask).save(str(img_out_dir / "5_raw_mask.png"))

    # ── 6. Save summary panel ──
    panel = create_summary_panel(img_rgb, overlay, conf_vis, pred_mask,
                                  class_names, num_classes)
    panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(img_out_dir / "6_summary_panel.jpg"), panel_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 90])

    # ── 7. Save legend ──
    legend = create_legend(class_names, num_classes)
    legend_bgr = cv2.cvtColor(legend, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(img_out_dir / "7_legend.jpg"), legend_bgr)

    # ── Build confidence report ──
    unique, counts = np.unique(pred_mask, return_counts=True)
    total = pred_mask.size

    report = {
        "image": Path(img_path).name,
        "width": w,
        "height": h,
        "mean_confidence": round(float(confidence.mean()), 4),
        "min_confidence": round(float(confidence.min()), 4),
        "num_classes_detected": len([u for u in unique if u > 0 and u not in EXCLUDED_CLASSES]),
        "fg_ratio": round(float(np.isin(pred_mask, [c for c in range(1, num_classes) if c not in EXCLUDED_CLASSES]).sum() / total), 4),
    }

    # Per-class stats (excluded classes omitted from report)
    for cls_id in range(num_classes):
        if cls_id in EXCLUDED_CLASSES:
            continue
        name = class_names.get(cls_id, f"class_{cls_id}")
        px = int((pred_mask == cls_id).sum())
        cls_conf = confidence[pred_mask == cls_id]
        report[f"{name}_pixels"] = px
        report[f"{name}_pct"] = round(px / total * 100, 2)
        report[f"{name}_mean_conf"] = round(float(cls_conf.mean()), 4) if len(cls_conf) > 0 else 0.0

    return report


def main():
    parser = argparse.ArgumentParser(description="Inference for thin-line segmentation")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to single image or folder of images")
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                        help="Model checkpoint path")
    parser.add_argument("--output", "-o", type=str, default=DEFAULT_OUTPUT,
                        help="Output directory")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Override number of classes (auto-detected if omitted)")
    parser.add_argument("--no_tta", action="store_true",
                        help="Disable test-time augmentation (faster)")
    args = parser.parse_args()

    # Resolve checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        if os.path.exists(DEFAULT_CHECKPOINT):
            ckpt_path = DEFAULT_CHECKPOINT
        elif os.path.exists(FALLBACK_CHECKPOINT):
            ckpt_path = FALLBACK_CHECKPOINT
            logger.info("Using fallback (original) checkpoint")
        else:
            logger.error("No checkpoint found! Train the model first or specify --checkpoint")
            sys.exit(1)

    # Load model
    model, num_classes = load_model(ckpt_path, args.num_classes)
    class_names = load_class_names(CLASSES_JSON, num_classes)
    transform = get_transform()

    logger.info(f"Classes: {class_names}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"TTA: {'disabled' if args.no_tta else 'enabled'}")

    # Resolve input (single file or folder)
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = sorted([f for f in input_path.iterdir()
                              if f.suffix.lower() in VALID_EXTENSIONS])
    else:
        logger.error(f"Input not found: {input_path}")
        sys.exit(1)

    logger.info(f"Images to process: {len(image_paths)}")

    # Setup output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all images
    all_reports = []
    for img_path in tqdm(image_paths, desc="Processing"):
        report = process_single_image(
            model, img_path, output_dir, class_names, num_classes,
            transform, use_tta=not args.no_tta,
        )
        if report:
            all_reports.append(report)

    # Save combined report
    if all_reports:
        report_path = output_dir / "inference_report.csv"
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_reports[0].keys())
            writer.writeheader()
            writer.writerows(all_reports)

        # Console summary
        logger.info(f"\n{'=' * 60}")
        logger.info("INFERENCE COMPLETE")
        logger.info(f"{'=' * 60}")
        logger.info(f"  Images processed: {len(all_reports)}")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Report saved:     {report_path}")

        avg_conf = np.mean([r["mean_confidence"] for r in all_reports])
        avg_classes = np.mean([r["num_classes_detected"] for r in all_reports])
        logger.info(f"  Avg confidence:   {avg_conf:.4f}")
        logger.info(f"  Avg classes/img:  {avg_classes:.1f}")

        logger.info(f"\nPer-image output structure:")
        logger.info(f"  <image_name>/")
        logger.info(f"    1_original.jpg              - Input image")
        logger.info(f"    2_prediction_overlay.jpg     - All classes color-coded")
        logger.info(f"    3_class_XX_<name>.jpg        - Per-class mask views")
        logger.info(f"    4_confidence_map.jpg         - Confidence heatmap")
        logger.info(f"    5_raw_mask.png               - Raw prediction mask (class IDs)")
        logger.info(f"    6_summary_panel.jpg          - 2x2 summary panel")
        logger.info(f"    7_legend.jpg                 - Color legend")
        logger.info(f"    inference_report.csv         - Combined stats for all images")


if __name__ == "__main__":
    main()
