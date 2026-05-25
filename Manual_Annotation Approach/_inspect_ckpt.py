"""Inspect existing checkpoint to understand model architecture and state."""
import torch
import sys

ckpt_path = r"C:\Users\Admin\line_detection\Approach 2\Segmentation_Deeplab_models_checkpoints\best_model.pth"

print("Loading checkpoint...")
try:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
except TypeError:
    ckpt = torch.load(ckpt_path, map_location='cpu')

if isinstance(ckpt, dict):
    print(f"Checkpoint type: dict with keys: {list(ckpt.keys())}")
    if 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
        print(f"  Epoch: {ckpt.get('epoch', '?')}")
        print(f"  Best mIoU: {ckpt.get('best_miou', '?')}")
    else:
        sd = ckpt
        print("  Raw state_dict (no wrapper)")
else:
    print(f"Checkpoint type: {type(ckpt)}")
    sd = ckpt

# Analyze state dict
print(f"\nTotal parameters: {len(sd)}")

# Group by top-level module
modules = {}
for k in sd.keys():
    top = k.split('.')[0]
    if top not in modules:
        modules[top] = []
    modules[top].append(k)

print(f"\nTop-level modules ({len(modules)}):")
for m, keys in modules.items():
    total_params = sum(sd[k].numel() for k in keys)
    print(f"  {m}: {len(keys)} tensors, {total_params:,} params")

# Check final classification layer
print("\nFinal layer keys (segmentation head):")
for k in sd.keys():
    if 'segmentation' in k.lower() or 'classifier' in k.lower() or 'final' in k.lower():
        print(f"  {k}: shape={sd[k].shape}")

# Check how many classes
for k in sd.keys():
    if sd[k].dim() >= 2:
        if 'segmentation_head' in k and 'weight' in k:
            print(f"\nSegmentation head output: {k} -> {sd[k].shape}")
            print(f"  => num_classes = {sd[k].shape[0]}")

# Check encoder name hints
print("\nSample encoder keys (first 5):")
encoder_keys = [k for k in sd.keys() if 'encoder' in k]
for k in encoder_keys[:5]:
    print(f"  {k}: {sd[k].shape}")

print("\nSample decoder keys (first 5):")
decoder_keys = [k for k in sd.keys() if 'decoder' in k]
for k in decoder_keys[:5]:
    print(f"  {k}: {sd[k].shape}")
