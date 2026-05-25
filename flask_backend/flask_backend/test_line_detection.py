#!/usr/bin/env python3
"""
Test Script for Line Detection Inference
-----------------------------------------
Tests the line detection inference module independently.

Usage:
    python test_line_detection.py --image path/to/image.png --output output_dir/
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add flask_backend to path
sys.path.insert(0, str(Path(__file__).parent))

from inference.line_detection_inference import LineDetectionInference
from config import Config


def test_line_detection(image_path: str, output_dir: str, use_tta: bool = False):
    """
    Test line detection on a single image.

    Args:
        image_path: Path to input image
        output_dir: Output directory
        use_tta: Use test-time augmentation
    """
    print("=" * 60)
    print("Line Detection Inference Test")
    print("=" * 60)

    # Check model exists
    if not os.path.exists(Config.LINE_DETECTION_MODEL_PATH):
        print(f"ERROR: Model not found at {Config.LINE_DETECTION_MODEL_PATH}")
        return

    # Check input image exists
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Model: {Config.LINE_DETECTION_MODEL_PATH}")
    print(f"Device: {Config.DEVICE}")
    print(f"Input: {image_path}")
    print(f"Output: {output_dir}")
    print(f"TTA: {use_tta}")
    print("-" * 60)

    # Initialize inference
    print("Loading model...")
    start_time = time.time()

    inference = LineDetectionInference(
        model_path=Config.LINE_DETECTION_MODEL_PATH,
        num_classes=Config.NUM_CLASSES,
        backbone=Config.BACKBONE,
        device=Config.DEVICE,
        tile_size=Config.TILE_SIZE,
        overlap=Config.OVERLAP,
        class_names=Config.CLASS_NAMES,
        class_colors=Config.CLASS_COLORS_RGB
    )

    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")

    if inference.model is None:
        print("ERROR: Failed to load model")
        return

    # Process image
    print("\nProcessing image...")
    result = inference.process_image(
        image_path=image_path,
        output_dir=output_dir,
        use_tta=use_tta
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if result['success']:
        print(f"Status: SUCCESS")
        print(f"\nPaths:")
        for key, path in result['paths'].items():
            print(f"  {key}: {path}")

        print(f"\nConfidence Report:")
        conf = result['confidence_report']
        print(f"  Overall: {conf['overall_confidence']:.3f}")
        print(f"  Per-class:")
        for class_name, score in conf['per_class'].items():
            count = conf['class_counts'].get(class_name, 0)
            if score > 0:
                print(f"    {class_name}: {score:.3f} ({count} pixels)")

        print(f"\nLegend ({len(result['legend'])} classes detected):")
        for item in result['legend'][:5]:
            print(f"  {item['class']}: {item['count']} pixels, {item['avg_confidence']:.3f} conf")
        if len(result['legend']) > 5:
            print(f"  ... and {len(result['legend']) - 5} more")

        print(f"\nTimings:")
        print(f"  Inference: {result['timings']['inference']:.2f}s")
        print(f"  Total: {result['timings']['total']:.2f}s")
    else:
        print(f"Status: FAILED")
        print(f"Error: {result.get('error', 'Unknown')}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test line detection inference")
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, default='test_output', help='Output directory')
    parser.add_argument('--tta', action='store_true', help='Use test-time augmentation')

    args = parser.parse_args()
    test_line_detection(args.image, args.output, args.tta)


if __name__ == '__main__':
    main()
