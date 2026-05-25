#!/usr/bin/env python3
"""
Test Script for Combined Pipeline
----------------------------------
Tests the full pipeline (filter then detect) independently.

Usage:
    python test_combined.py --image path/to/image.png --output output_dir/
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add flask_backend to path
sys.path.insert(0, str(Path(__file__).parent))

from inference.line_detection_inference import LineDetectionInference
from inference.filter_out_inference import FilterOutInference
from config import Config


def test_combined(image_path: str, output_dir: str, use_tta: bool = False):
    """
    Test combined pipeline on a single image.

    Args:
        image_path: Path to input image
        output_dir: Output directory
        use_tta: Use test-time augmentation
    """
    print("=" * 60)
    print("Combined Pipeline Test (Filter -> Detect)")
    print("=" * 60)

    # Check models exist
    if not os.path.exists(Config.LINE_DETECTION_MODEL_PATH):
        print(f"ERROR: Line detection model not found")
        return

    if not os.path.exists(Config.FILTER_MODEL_PATH):
        print(f"ERROR: Filter model not found")
        return

    # Check input image exists
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Input: {image_path}")
    print(f"Output: {output_dir}")
    print("-" * 60)

    total_start = time.time()

    # Initialize filter inference
    print("\n1. Loading filter model...")
    filter_inference = FilterOutInference(
        model_path=Config.FILTER_MODEL_PATH,
        conf_threshold=Config.FILTER_CONF_THRESHOLD,
        max_imgsz=Config.MAX_YOLO_IMGSZ
    )

    if filter_inference.model is None:
        print("ERROR: Failed to load filter model")
        return

    # Initialize line detection inference
    print("2. Loading line detection model...")
    detection_inference = LineDetectionInference(
        model_path=Config.LINE_DETECTION_MODEL_PATH,
        num_classes=Config.NUM_CLASSES,
        backbone=Config.BACKBONE,
        device=Config.DEVICE,
        tile_size=Config.TILE_SIZE,
        overlap=Config.OVERLAP,
        temperature=Config.TEMPERATURE,
        class_names=Config.CLASS_NAMES,
        class_colors=Config.CLASS_COLORS_RGB
    )

    if detection_inference.model is None:
        print("ERROR: Failed to load line detection model")
        return

    # Step 1: Filter-out
    print("\n3. Running filter-out...")
    filter_result = filter_inference.process_image(
        image_path=image_path,
        output_dir=output_dir
    )

    if not filter_result['success']:
        print(f"Filter failed: {filter_result.get('error')}")
        return

    filter_stats = filter_result['statistics']
    print(f"   Removed: {filter_stats['removal_percentage']:.1f}% ({filter_stats['removed_pixel_count']:,} pixels)")

    # Get filtered image for detection
    filtered_image = filter_result['_filtered_image']

    # Step 2: Save filtered image temporarily, then run line detection
    print("4. Running line detection on filtered image...")
    from PIL import Image
    import numpy as np

    filename = os.path.basename(image_path)
    basename = os.path.splitext(filename)[0]

    # Save filtered image to temp file for process_image()
    temp_filtered_path = os.path.join(output_dir, f'_temp_filtered_{filename}')
    Image.fromarray(filtered_image).save(temp_filtered_path)

    # Run detection using process_image (matches inference_step3.py)
    detection_result = detection_inference.process_image(
        image_path=temp_filtered_path,
        output_dir=output_dir,
        use_tta=use_tta,
        enhance_contrast=True
    )

    # Clean up temp file
    if os.path.exists(temp_filtered_path):
        os.remove(temp_filtered_path)

    total_time = time.time() - total_start

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Filter removal: {filter_stats['removal_percentage']:.1f}%")

    if detection_result.get('success', False):
        print(f"Detection stats: {detection_result.get('stats', {})}")
        print(f"Timings: {detection_result.get('timings', {})}")
    else:
        print(f"Detection failed: {detection_result.get('error', 'Unknown error')}")

    print(f"Total time: {total_time:.2f}s")

    print(f"\nOutput files in: {output_dir}")
    print("  - filtered_images/: Filtered images")
    print("  - masks/: Segmentation masks")
    print("  - binary_masks/: Per-class binary masks")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test combined pipeline")
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, default='test_combined_output', help='Output directory')
    parser.add_argument('--tta', action='store_true', help='Use test-time augmentation')

    args = parser.parse_args()
    test_combined(args.image, args.output, args.tta)


if __name__ == '__main__':
    main()
