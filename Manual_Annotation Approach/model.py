

import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)


class EdgeHead(nn.Module):
    
    def __init__(self, in_channels=256, mid_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels // 2, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels // 2)
        self.conv_out = nn.Conv2d(mid_channels // 2, 1, 1)
        self.relu = nn.ReLU(inplace=True)

        # Initialize with small weights so edge head doesn't
        # destabilize early training
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv_out(x)
        return x


class DeepLabV3PlusEdge(nn.Module):
    

    def __init__(self, num_classes=10, backbone="resnet50", use_edge_head=True):
        super().__init__()

        # Build the base SMP model
        self.base_model = smp.DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )

        self.use_edge_head = use_edge_head

        if use_edge_head:
            # The SMP decoder outputs 256 channels before the seg head
            self.edge_head = EdgeHead(in_channels=256, mid_channels=64)
        else:
            self.edge_head = None

    def forward(self, x):
        
        input_size = x.shape[2:]  # (H, W)

        # Use SMP's internal encoder → decoder flow
        features = self.base_model.encoder(x)  # list of feature maps
        decoder_output = self.base_model.decoder(features)  # (B, 256, H/4, W/4)

        # Segmentation head (Conv1x1 + Upsample built into SMP head)
        seg_logits = self.base_model.segmentation_head(decoder_output)

        # Edge head (branches from the same decoder features)
        edge_logits = None
        if self.use_edge_head and self.edge_head is not None:
            edge_logits = self.edge_head(decoder_output)
            # Upsample to match input size
            edge_logits = F.interpolate(
                edge_logits, size=input_size, mode="bilinear", align_corners=False
            )

        return seg_logits, edge_logits

    # ── Checkpoint Loading ────────────────────────────────────────────

    def load_pretrained_checkpoint(self, ckpt_path, old_num_classes=13, device="cpu"):
        
        logger.info(f"Loading checkpoint: {ckpt_path}")

        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=device)

        # Extract state dict
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            old_sd = ckpt["model_state_dict"]
            epoch = ckpt.get("epoch", "?")
            best_miou = ckpt.get("best_miou", "?")
            logger.info(f"  Checkpoint epoch: {epoch}, best mIoU: {best_miou}")
        else:
            old_sd = ckpt
            logger.info("  Raw state_dict loaded (no metadata)")

        new_sd = self.base_model.state_dict()
        new_num_classes = self.base_model.segmentation_head[0].out_channels

        loaded = 0
        skipped = 0
        adapted = 0

        # Build mapping: strip 'base_model.' prefix if needed
        filtered_sd = OrderedDict()

        for key, value in old_sd.items():
            if key in new_sd:
                if "segmentation_head" in key and old_num_classes != new_num_classes:
                    # Handle class count mismatch
                    if "weight" in key:
                        # Old: (13, 256, 1, 1), New: (10, 256, 1, 1)
                        # Copy weights for classes that still exist
                        # Classes 0 (BG) maps to 0, rest need careful mapping
                        old_w = value  # (13, 256, 1, 1)
                        new_w = new_sd[key].clone()  # (10, 256, 1, 1)

                        # Copy background weights (class 0)
                        new_w[0] = old_w[0]

                        # Copy the first min(old-1, new-1) foreground classes
                        n_copy = min(old_num_classes - 1, new_num_classes - 1)
                        new_w[1:1 + n_copy] = old_w[1:1 + n_copy]

                        filtered_sd[key] = new_w
                        adapted += 1
                        logger.info(f"  ADAPTED {key}: {old_w.shape} -> {new_w.shape} "
                                    f"(copied {n_copy + 1}/{old_num_classes} classes)")

                    elif "bias" in key:
                        old_b = value  # (13,)
                        new_b = new_sd[key].clone()  # (10,)
                        new_b[0] = old_b[0]
                        n_copy = min(old_num_classes - 1, new_num_classes - 1)
                        new_b[1:1 + n_copy] = old_b[1:1 + n_copy]
                        filtered_sd[key] = new_b
                        adapted += 1
                    else:
                        filtered_sd[key] = value
                        loaded += 1

                elif new_sd[key].shape == value.shape:
                    filtered_sd[key] = value
                    loaded += 1
                else:
                    logger.warning(f"  SKIPPED {key}: shape mismatch "
                                   f"{value.shape} vs {new_sd[key].shape}")
                    skipped += 1
            else:
                logger.debug(f"  IGNORED {key}: not in new model")
                skipped += 1

        # Load filtered weights
        result = self.base_model.load_state_dict(filtered_sd, strict=False)

        missing = len(result.missing_keys)
        unexpected = len(result.unexpected_keys)

        stats = {
            "loaded": loaded,
            "adapted": adapted,
            "skipped": skipped,
            "missing_in_ckpt": missing,
            "unexpected": unexpected,
        }

        logger.info(f"  Checkpoint loaded: {loaded} exact, {adapted} adapted, "
                     f"{skipped} skipped, {missing} missing (new layers)")

        if result.missing_keys:
            logger.debug(f"  Missing keys (expected for new heads): "
                         f"{result.missing_keys[:5]}...")

        return stats

    # ── Freeze / Unfreeze Utilities ───────────────────────────────────

    def freeze_encoder(self):
        """Freeze all encoder (backbone) parameters for Phase 1 training."""
        count = 0
        for param in self.base_model.encoder.parameters():
            param.requires_grad = False
            count += 1
        logger.info(f"Encoder frozen: {count} parameters")

    def unfreeze_encoder(self):
        """Unfreeze encoder for Phase 2 fine-tuning."""
        count = 0
        for param in self.base_model.encoder.parameters():
            param.requires_grad = True
            count += 1
        logger.info(f"Encoder unfrozen: {count} parameters")

    def freeze_decoder(self):
        """Freeze decoder parameters."""
        for param in self.base_model.decoder.parameters():
            param.requires_grad = False
        logger.info("Decoder frozen")

    def unfreeze_all(self):
        """Unfreeze everything."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All parameters unfrozen")

    def get_param_groups(self, lr_backbone=1e-5, lr_decoder=5e-5, lr_heads=1e-4):
        
        backbone_params = []
        decoder_params = []
        head_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "encoder" in name:
                backbone_params.append(param)
            elif "decoder" in name:
                decoder_params.append(param)
            else:
                head_params.append(param)

        groups = [
            {"params": backbone_params, "lr": lr_backbone, "name": "backbone"},
            {"params": decoder_params, "lr": lr_decoder, "name": "decoder"},
            {"params": head_params, "lr": lr_heads, "name": "heads"},
        ]

        logger.info(f"Parameter groups: backbone={len(backbone_params)} params @ lr={lr_backbone}, "
                     f"decoder={len(decoder_params)} @ lr={lr_decoder}, "
                     f"heads={len(head_params)} @ lr={lr_heads}")

        return groups

    def count_parameters(self):
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {"total": total, "trainable": trainable, "frozen": frozen}

    def summary(self):
        """Print a concise model summary."""
        counts = self.count_parameters()
        logger.info(f"Model Summary:")
        logger.info(f"  Total params:     {counts['total']:,}")
        logger.info(f"  Trainable params: {counts['trainable']:,}")
        logger.info(f"  Frozen params:    {counts['frozen']:,}")
        logger.info(f"  Edge head:        {'enabled' if self.use_edge_head else 'disabled'}")
        return counts
