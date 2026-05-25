#!/usr/bin/env python3
"""
Test Script for Filter-Out Inference
-------------------------------------
Tests the filter-out inference module independently.

Usage:
    python test_filter_out.py --image path/to/image.png --output output_dir/
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add flask_backend to path
sys.path.insert(0, str(Path(__file__).parent))

from inference.filter_out_inference import FilterOutInference
from config import Config


def test_filter_out(image_path: str, output_dir: str, conf_threshold: float = None):
    """
    Test filter-out on a single image.

    Args:
        image_path: Path to input image
        output_dir: Output directory
        conf_threshold: Confidence threshold override
    """
    print("=" * 60)
    print("Filter-Out Inference Test")
    print("=" * 60)

    # Check model exists
    if not os.path.exists(Config.FILTER_MODEL_PATH):
        print(f"ERROR: Model not found at {Config.FILTER_MODEL_PATH}")
        return

    # Check input image exists
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    conf_thresh = conf_threshold or Config.FILTER_CONF_THRESHOLD

    print(f"Model: {Config.FILTER_MODEL_PATH}")
    print(f"Input: {image_path}")
    print(f"Output: {output_dir}")
    print(f"Confidence threshold: {conf_thresh}")
    print("-" * 60)

    # Initialize inference
    print("Loading model...")
    start_time = time.time()

    inference = FilterOutInference(
        model_path=Config.FILTER_MODEL_PATH,
        conf_threshold=conf_thresh,
        max_imgsz=Config.MAX_YOLO_IMGSZ
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
        output_dir=output_dir
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if result['success']:
        print(f"Status: SUCCESS")
        print(f"\nPaths:")
        for key, path in result['paths'].items():
            if path:
                print(f"  {key}: {path}")

        stats = result['statistics']
        print(f"\nStatistics:")
        print(f"  Filter Score: {stats['overall_filter_score']:.3f}")
        print(f"  Removed Pixels: {stats['removed_pixel_count']:,}")
        print(f"  Retained Pixels: {stats['retained_pixel_count']:,}")
        print(f"  Removal Percentage: {stats['removal_percentage']:.2f}%")
        print(f"  Detections: {stats['detection_count']}")

        if stats['detections']:
            print(f"\nDetected Regions:")
            for i, det in enumerate(stats['detections'][:5]):
                print(f"  {i+1}. Class {det['class_id']}: conf={det['confidence']:.3f}, area={det['area']:,}")

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
    parser = argparse.ArgumentParser(description="Test filter-out inference")
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, default='test_filter_output', help='Output directory')
    parser.add_argument('--threshold', type=float, help='Confidence threshold')

    args = parser.parse_args()
    test_filter_out(args.image, args.output, args.threshold)


if __name__ == '__main__':
    main()
