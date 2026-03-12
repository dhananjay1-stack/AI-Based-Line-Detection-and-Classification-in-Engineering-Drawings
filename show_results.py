import torch
import json
import pandas as pd
import os
import glob
from pathlib import Path

# --- CONFIGURATION ---
SEG_MODEL_PATH = "./output/models/best_model.pth"
SEG_EVAL_JSON  = "./output/eval/eval_seg.json"
YOLO_CSV_DIR   = "./yolo_runs" # Root folder where YOLO logs are saved
# ---------------------

def check_seg_training():
    print(f"\n{'='*20} SEGMENTATION TRAINING {'='*20}")
    if not os.path.exists(SEG_MODEL_PATH):
        print(f"❌ No model found at {SEG_MODEL_PATH}")
        return

    try:
        # Load safely
        try:
            checkpoint = torch.load(SEG_MODEL_PATH, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(SEG_MODEL_PATH, map_location='cpu')

        if isinstance(checkpoint, dict) and 'best_miou' in checkpoint:
            epoch = checkpoint.get('epoch', 'Unknown')
            miou = checkpoint.get('best_miou', 0.0)
            print(f"✅ Best Model saved at Epoch: {epoch}")
            print(f"🏆 Best Validation mIoU:     {miou:.4f} ({(miou*100):.2f}%)")
        else:
            print("⚠️  Checkpoint exists but contains no metadata (Legacy format).")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")

def check_seg_evaluation():
    print(f"\n{'='*20} SEGMENTATION EVALUATION {'='*20}")
    if not os.path.exists(SEG_EVAL_JSON):
        print(f"ℹ️  No evaluation file found at {SEG_EVAL_JSON}")
        print("   (Run 'python part2_seg.py eval ...' to generate this)")
        return

    with open(SEG_EVAL_JSON, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame for pretty printing
    df = pd.DataFrame(data).T
    if 'mIoU' in df.columns:
        df = df.sort_values(by='mIoU', ascending=False)
        print(df[['mIoU', 'F1']].to_string())
        print("-" * 40)
        print(f"🌍 Average mIoU: {df['mIoU'].mean():.4f}")
    else:
        print(data)

def check_yolo_results():
    print(f"\n{'='*20} YOLO DETECTION RESULTS {'='*20}")
    # Find the results.csv file (it might be in yolo_runs/core_run/results.csv)
    csv_files = glob.glob(os.path.join(YOLO_CSV_DIR, "**", "results.csv"), recursive=True)
    
    if not csv_files:
        print("❌ No YOLO 'results.csv' logs found.")
        return

    # Use the most recently modified CSV
    latest_csv = max(csv_files, key=os.path.getmtime)
    print(f"📂 Reading log: {latest_csv}")

    try:
        df = pd.read_csv(latest_csv)
        # Strip whitespace from column names
        df.columns = [c.strip() for c in df.columns]
        
        # Find best mAP50-95
        if 'metrics/mAP50-95(B)' in df.columns:
            best_idx = df['metrics/mAP50-95(B)'].idxmax()
            best_row = df.iloc[best_idx]
            
            print(f"✅ Best Epoch: {best_row['epoch']}")
            print(f"🏆 Best mAP@50-95: {best_row['metrics/mAP50-95(B)']:.4f}")
            print(f"   Best mAP@50:    {best_row['metrics/mAP50(B)']:.4f}")
            print(f"   Precision:      {best_row['metrics/precision(B)']:.4f}")
            print(f"   Recall:         {best_row['metrics/recall(B)']:.4f}")
        else:
            print("⚠️  Could not find mAP columns in CSV.")
            print(f"   Found columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Error parsing YOLO CSV: {e}")

if __name__ == "__main__":
    check_seg_training()
    check_seg_evaluation()
    check_yolo_results()
    print("\n" + "="*65)