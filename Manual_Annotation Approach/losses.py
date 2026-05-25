

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class segmentation.
    
    Focuses learning on hard-to-classify pixels (thin lines vs background),
    reducing the contribution of easy background pixels that dominate the image.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma=2.0, alpha=None, ignore_index=-100, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Can be a tensor of per-class weights
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) raw model output
            targets: (B, H, W) integer class labels
        """
        num_classes = logits.shape[1]
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.alpha,
            ignore_index=self.ignore_index,
            reduction="none",
        )  # (B, H, W)

        # p_t = probability of the true class
        log_pt = -ce_loss
        pt = torch.exp(log_pt)

        focal_term = (1.0 - pt) ** self.gamma
        loss = focal_term * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for multi-class segmentation.
    
    Directly optimizes overlap between prediction and ground truth,
    making it less sensitive to class imbalance than CE alone.
    """

    def __init__(self, smooth=1.0, ignore_index=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W)
            targets: (B, H, W)
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # One-hot encode targets
        targets_oh = F.one_hot(targets.long(), num_classes)  # (B, H, W, C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Compute per-class dice
        dims = (0, 2, 3)  # sum over batch, H, W
        intersection = (probs * targets_oh).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets_oh.sum(dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Skip background (index 0) if desired — but we keep it for stability
        loss = 1.0 - dice_score.mean()
        return loss


class EdgeLoss(nn.Module):
    """
    Binary cross-entropy + Dice loss for edge/centerline prediction.
    
    The edge head predicts a single-channel map of 1px skeleton structures.
    We use BCE + Dice to handle the extreme foreground/background imbalance
    (skeletons are ~0.01% of pixels).
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, pos_weight=10.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        # pos_weight amplifies the loss for foreground skeleton pixels
        self.bce_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

    def forward(self, edge_logits, edge_targets):
        """
        Args:
            edge_logits: (B, 1, H, W) raw logits from edge head
            edge_targets: (B, 1, H, W) binary skeleton mask (0 or 1)
        """
        # Move pos_weight to correct device
        if self.bce_fn.pos_weight.device != edge_logits.device:
            self.bce_fn.pos_weight = self.bce_fn.pos_weight.to(edge_logits.device)

        bce = self.bce_fn(edge_logits, edge_targets.float())

        # Sigmoid dice
        probs = torch.sigmoid(edge_logits)
        smooth = 1.0
        dims = (0, 2, 3)
        intersection = (probs * edge_targets.float()).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + edge_targets.float().sum(dim=dims)
        dice = 1.0 - (2.0 * intersection + smooth) / (cardinality + smooth)
        dice = dice.mean()

        return self.bce_weight * bce + self.dice_weight * dice


class CombinedLoss(nn.Module):
    """
    Full combined loss for thin-line segmentation fine-tuning.
    
    L_total = w_ce * CE + w_dice * Dice + w_focal * Focal + w_edge * EdgeLoss
    
    Class weights are applied to CE and Focal to handle imbalance.
    Edge loss teaches the model to preserve 1-pixel structures.
    """

    def __init__(
        self,
        num_classes,
        class_weights=None,
        ce_weight=1.0,
        dice_weight=1.0,
        focal_weight=0.5,
        edge_weight=0.5,
        focal_gamma=2.0,
        use_edge=True,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.edge_weight = edge_weight
        self.use_edge = use_edge

        # Class-weighted CE
        self.ce_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            reduction="mean",
        )

        # Dice
        self.dice_fn = DiceLoss(smooth=1.0)

        # Focal
        self.focal_fn = FocalLoss(
            gamma=focal_gamma,
            alpha=class_weights,
            reduction="mean",
        ) if focal_weight > 0 else None

        # Edge
        self.edge_fn = EdgeLoss(
            bce_weight=1.0,
            dice_weight=1.0,
            pos_weight=10.0,
        ) if use_edge else None

    def forward(self, seg_logits, targets, edge_logits=None, edge_targets=None):
        """
        Args:
            seg_logits: (B, C, H, W) main segmentation output
            targets: (B, H, W) class labels
            edge_logits: (B, 1, H, W) edge head output (optional)
            edge_targets: (B, 1, H, W) skeleton ground truth (optional)
        
        Returns:
            total_loss, loss_dict with individual components
        """
        loss_dict = {}

        # Segmentation losses
        l_ce = self.ce_fn(seg_logits, targets)
        loss_dict["ce"] = l_ce.item()

        l_dice = self.dice_fn(seg_logits, targets)
        loss_dict["dice"] = l_dice.item()

        total = self.ce_weight * l_ce + self.dice_weight * l_dice

        if self.focal_fn is not None and self.focal_weight > 0:
            l_focal = self.focal_fn(seg_logits, targets)
            loss_dict["focal"] = l_focal.item()
            total = total + self.focal_weight * l_focal

        # Edge loss
        if (
            self.use_edge
            and self.edge_fn is not None
            and edge_logits is not None
            and edge_targets is not None
        ):
            l_edge = self.edge_fn(edge_logits, edge_targets)
            loss_dict["edge"] = l_edge.item()
            total = total + self.edge_weight * l_edge

        loss_dict["total"] = total.item()
        return total, loss_dict
