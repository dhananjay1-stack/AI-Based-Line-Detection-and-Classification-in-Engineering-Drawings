"""Smoke test: model init, checkpoint loading, forward pass, loss computation."""
import sys
sys.path.insert(0, r"C:\Users\Admin\line_detection\Approach 2")

import torch
import numpy as np

from config import TrainConfig, ClassInfo
from model import DeepLabV3PlusEdge
from losses import CombinedLoss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

cfg = TrainConfig()

# 1. Build model
print("\n[1] Building model...")
model = DeepLabV3PlusEdge(
    num_classes=cfg.num_classes,
    backbone=cfg.backbone,
    use_edge_head=cfg.use_edge_head,
)

# 2. Load checkpoint
print("\n[2] Loading checkpoint (13 -> 10 classes)...")
stats = model.load_pretrained_checkpoint(
    cfg.checkpoint_path,
    old_num_classes=cfg.old_num_classes,
    device="cpu",
)
print(f"  Stats: {stats}")

model = model.to(DEVICE)

# 3. Summary
print("\n[3] Model summary:")
counts = model.summary()

# 4. Phase 1: freeze encoder
print("\n[4] Phase 1: Freeze encoder...")
model.freeze_encoder()
counts = model.count_parameters()
print(f"  Trainable: {counts['trainable']:,} / {counts['total']:,}")

# 5. Forward pass
print("\n[5] Forward pass test...")
dummy_img = torch.randn(2, 3, 512, 512).to(DEVICE)
with torch.no_grad():
    seg_out, edge_out = model(dummy_img)
print(f"  seg_out shape: {seg_out.shape}")
print(f"  edge_out shape: {edge_out.shape if edge_out is not None else 'None'}")
assert seg_out.shape == (2, 10, 512, 512), f"Bad seg shape: {seg_out.shape}"
assert edge_out.shape == (2, 1, 512, 512), f"Bad edge shape: {edge_out.shape}"
print("  Shapes OK!")

# 6. Loss computation
print("\n[6] Loss computation test...")
class_weights = torch.ones(cfg.num_classes).to(DEVICE)
criterion = CombinedLoss(
    num_classes=cfg.num_classes,
    class_weights=class_weights,
    ce_weight=1.0,
    dice_weight=1.0,
    focal_weight=0.5,
    edge_weight=0.5,
    focal_gamma=2.0,
    use_edge=True,
)

dummy_mask = torch.randint(0, 10, (2, 512, 512)).to(DEVICE)
dummy_edge = torch.zeros(2, 1, 512, 512).to(DEVICE)

seg_out_grad = seg_out.detach().requires_grad_(True)
total_loss, loss_dict = criterion(seg_out_grad, dummy_mask, edge_out, dummy_edge)
print(f"  Loss dict: {loss_dict}")
print(f"  Total loss: {total_loss.item():.4f}")

# 7. Phase 2: unfreeze + param groups
print("\n[7] Phase 2: Unfreeze + differential LR...")
model.unfreeze_all()
param_groups = model.get_param_groups(lr_backbone=1e-5, lr_decoder=5e-5, lr_heads=1e-4)
for g in param_groups:
    print(f"  {g['name']}: {len(g['params'])} params @ lr={g['lr']}")

# 8. Class info
print("\n[8] Class info:")
ci = ClassInfo.from_json(cfg.classes_json)
for cid, name in sorted(ci.id_to_name.items()):
    print(f"  {cid}: {name}")

print("\n" + "=" * 50)
print("ALL SMOKE TESTS PASSED!")
print("=" * 50)
