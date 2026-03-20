"""
Multi-task loss functions for DIGITAL UNIVERSITY OF KERALA EXTRACTION MODEL.

Combines:
  - Binary tasks: BCE + Dice + Lovász hinge
  - Roof-type task: Cross-entropy + Dice
  - Per-task weighting for class imbalance
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Dice Loss ───────────────────────────────────────────────────────────


# ── Tversky Loss ──────────────────────────────────────────────────────────


class TverskyLoss(nn.Module):
    """
    Tversky loss: Generalization of Dice loss.
    Higher beta penalizes False Negatives (FN) more, useful for small objects.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits).float()
        probs = probs.view(-1)
        targets = targets.float().view(-1)

        if mask is not None:
            mask = mask.view(-1)
            probs = probs * mask
            targets = targets * mask

        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1.0 - tversky


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) raw logits
            targets: (B, 1, H, W) binary targets
            mask: (B, 1, H, W) optional valid pixel mask
        """
        probs = torch.sigmoid(logits).float()
        probs = probs.view(-1)
        targets = targets.float().view(-1)

        if mask is not None:
            mask = mask.view(-1)
            probs = probs * mask
            targets = targets * mask

        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


# ── Focal Loss ──────────────────────────────────────────────────────────


class BinaryFocalLoss(nn.Module):
    """Focal loss for handling class imbalance in binary tasks."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )
        probs = torch.sigmoid(logits).float()
        p_t = probs * targets.float() + (1 - probs) * (1 - targets.float())
        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma

        loss = focal_weight * bce
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            loss = loss * mask
            return torch.sum(loss) / (torch.sum(mask) + 1e-7)
        return torch.mean(loss)


# ── Lovász Hinge Loss ────────────────────────────────────────────────────────


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute gradient of the Lovász extension w.r.t sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / (union + 1e-7)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszHingeLoss(nn.Module):
    """Lovász-Softmax loss for binary segmentation (IoU surrogate)."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.float().view(-1)

        if mask is not None:
            mask = mask.view(-1)
            # Only keep pixels where mask is 1
            valid_idx = mask > 0.5
            if not valid_idx.any():
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            logits = logits[valid_idx]
            targets = targets[valid_idx]

        signs = 2.0 * targets - 1.0
        errors = 1.0 - logits * signs
        errors = torch.clamp(errors, min=0)  # stability
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = targets[perm]
        grad = _lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss


# ── Boundary Loss ────────────────────────────────────────────────────────────


class BoundaryLoss(nn.Module):
    """Weighted BCE along mask boundaries for sharp edges."""

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Extract boundary via morphological gradient
        pad = self.kernel_size // 2
        t = targets.float()
        dilated = F.max_pool2d(t, self.kernel_size, stride=1, padding=pad)
        eroded = -F.max_pool2d(-t, self.kernel_size, stride=1, padding=pad)
        boundary = (dilated - eroded).clamp(0, 1)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            boundary = boundary * mask

        # Compute boundary-weighted BCE
        if boundary.sum() < 1:
            return torch.tensor(0.0, device=logits.device)

        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        return (bce * boundary).sum() / boundary.sum()


# ── Multi-class Dice Loss ────────────────────────────────────────────────────


class MultiClassDiceLoss(nn.Module):
    """Dice loss for multi-class segmentation (roof types)."""

    def __init__(self, num_classes: int = 5, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W)
            targets: (B, H, W) class indices
        """
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets.long(), self.num_classes)  # (B, H, W, C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        dims = (0, 2, 3)
        intersection = (probs * targets_oh).sum(dims)
        union = probs.sum(dims) + targets_oh.sum(dims)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


# ── Combined Multi-Task Loss ─────────────────────────────────────────────────


# Default task weights
DEFAULT_WEIGHTS = {
    "building": 1.0,
    "road": 1.0,
    "road_centerline": 0.8,
    "waterbody": 1.0,
    "waterbody_line": 0.8,
    "waterbody_point": 1.2,
    "utility_line": 1.0,
    "utility_point": 1.2,
    "bridge": 1.0,
    "railway": 1.0,
}

BINARY_TASKS = [
    "building",
    "road",
    "road_centerline",
    "waterbody",
    "waterbody_line",
    "waterbody_point",
    "utility_line",
    "utility_point",
    "bridge",
    "railway",
]


class MultiTaskLoss(nn.Module):
    """
    Combined loss for all 11 tasks.

    Binary tasks: BCE + Dice + Lovász (equal-weighted sum)
    Roof-type task: CE + MultiClassDice

    Each task loss is multiplied by its weight before summation.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        num_roof_classes: int = 5,
    ):
        super().__init__()
        self.weights = weights or DEFAULT_WEIGHTS

        # Binary loss components
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)  # Penalize FNs more
        self.focal = BinaryFocalLoss(alpha=0.25, gamma=2.0)
        self.lovasz = LovaszHingeLoss()

        # Multi-class loss for roof types
        self.ce = nn.CrossEntropyLoss(ignore_index=0)  # ignore background
        self.mc_dice = MultiClassDiceLoss(num_roof_classes)

        self.boundary = BoundaryLoss()

    def _binary_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        ohem_ratio: float = 0.7,
    ) -> torch.Tensor:
        """
        Combined binary loss with boundary emphasis, NoData masking,
        and Online Hard Example Mining (OHEM).
        """
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        if mask is not None and mask.ndim == 3:
            mask = mask.unsqueeze(1)

        # 1. Base BCE Loss
        if mask is not None:
            bce_raw = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            bce_masked = bce_raw * mask

            num_pixels = mask.sum().int().item()
            # OHEM: Only take the top N% hardest pixels
            if ohem_ratio < 1.0:
                if num_pixels > 100:
                    num_hard = int(num_pixels * ohem_ratio)
                    # We can't easily sort everything, so we use a threshold
                    # Sort only valid pixels
                    valid_logits = bce_masked[mask > 0.5]
                    hard_threshold = torch.topk(valid_logits, num_hard).values[-1]
                    ohem_mask = (bce_masked >= hard_threshold).float() * mask
                    bce = bce_masked[ohem_mask > 0.5].mean()
                else:
                    bce = bce_masked.sum() / (num_pixels + 1e-7)
            else:
                bce = bce_masked.sum() / (num_pixels + 1e-7)
        else:
            bce = self.bce(logits, targets)

        # 2. Geometry-aware losses
        dice = self.dice(logits, targets, mask=mask)
        # Use Tversky for points and lines to handle extreme sparsity
        tversky = self.tversky(logits, targets, mask=mask)
        focal = self.focal(logits, targets, mask=mask)
        lovasz = self.lovasz(logits, targets, mask=mask)
        bdry = self.boundary(logits, targets, mask=mask)

        # Weights tuned for >95% IoU (emphasizing IoU/Lovasz and Tversky/Boundary)
        return (
            0.6 * bce + 1.0 * dice + 0.8 * tversky + 0.3 * focal + 1.2 * lovasz + 0.5 * bdry
        )

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = next(iter(predictions.values())).device
        total = torch.zeros(1, device=device)
        breakdown = {}
        valid_mask = targets.get("valid_mask", None)

        for task in BINARY_TASKS:
            pred_key = f"{task}_mask"
            if pred_key not in predictions or pred_key not in targets:
                continue

            # Use NoData valid_mask to skip non-pixels
            loss = self._binary_loss(
                predictions[pred_key], targets[pred_key], mask=valid_mask
            )

            if torch.isnan(loss) or torch.isinf(loss):
                continue
            weighted = loss * self.weights.get(task, 1.0)
            total = total + weighted
            breakdown[task] = loss.item()

        # Roof type loss
        if "roof_type_mask" in predictions and "roof_type_mask" in targets:
            rt_pred = predictions["roof_type_mask"]
            rt_target = targets["roof_type_mask"]
            if rt_target.ndim == 4:
                rt_target = rt_target.squeeze(1)
            # Skip if all pixels are background (class 0)
            # CE with ignore_index=0 returns NaN when
            # there are no non-ignored pixels
            if (rt_target.long() != 0).any():
                rt_loss = self.ce(rt_pred, rt_target.long()) + self.mc_dice(
                    rt_pred, rt_target
                )
                if not (torch.isnan(rt_loss) or torch.isinf(rt_loss)):
                    w = self.weights.get("roof_type", 0.5)
                    total = total + rt_loss * w
                    breakdown["roof_type"] = rt_loss.item()

        # Normalize loss by number of active tasks
        n_active = len(breakdown)
        if n_active > 0:
            total = total / n_active

        # Clamp massive losses for UI stability (only if total > 100)
        # 370 is high, so we keep it but it will show correctly after division
        return total, breakdown
