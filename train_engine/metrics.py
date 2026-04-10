"""
Per-task evaluation metrics: IoU, Dice/F1, pixel accuracy.
"""

from typing import Dict, List, Optional

import numpy as np
import torch


class TaskMetrics:
    """
    Accumulates predictions and targets for a single binary task
    and computes IoU, Dice, and pixel accuracy.
    """

    def __init__(self, name: str, threshold: float = 0.5):
        self.name = name
        self.threshold = threshold
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Update with a batch of predictions.

        Args:
            logits: (B, 1, H, W) raw logits
            targets: (B, 1, H, W) or (B, H, W) binary targets
            mask: (B, H, W) optional valid pixel mask
        """
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > self.threshold).float()
            if preds.ndim == 4:
                preds = preds.squeeze(1)
            if targets.ndim == 4:
                targets = targets.squeeze(1)
            targets = targets.float()

            if mask is not None:
                if mask.ndim == 4:
                    mask = mask.squeeze(1)
                mask = mask > 0.5
                # Apply mask to all components
                t_tp = (preds == 1) & (targets == 1) & mask
                t_fp = (preds == 1) & (targets == 0) & mask
                t_fn = (preds == 0) & (targets == 1) & mask
                t_tn = (preds == 0) & (targets == 0) & mask

                self.tp += t_tp.sum().item()
                self.fp += t_fp.sum().item()
                self.fn += t_fn.sum().item()
                self.tn += t_tn.sum().item()
            else:
                self.tp += ((preds == 1) & (targets == 1)).sum().item()
                self.fp += ((preds == 1) & (targets == 0)).sum().item()
                self.fn += ((preds == 0) & (targets == 1)).sum().item()
                self.tn += ((preds == 0) & (targets == 0)).sum().item()

    @property
    def iou(self) -> float:
        denom = self.tp + self.fp + self.fn
        return self.tp / (denom + 1e-8)

    @property
    def dice(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return (2 * self.tp) / (denom + 1e-8)

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / (denom + 1e-8)

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / (denom + 1e-8)

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / (total + 1e-8)

    def compute(self) -> Dict[str, float]:
        return {
            f"{self.name}_iou": self.iou,
            f"{self.name}_dice": self.dice,
            f"{self.name}_precision": self.precision,
            f"{self.name}_recall": self.recall,
            f"{self.name}_accuracy": self.accuracy,
        }

    def get_stats_tensor(self, device: str = None) -> torch.Tensor:
        device = device or "cuda"
        return torch.tensor(
            [self.tp, self.fp, self.fn, self.tn],
            dtype=torch.float64,
            device=device,
        )

    def set_stats_from_tensor(self, data: torch.Tensor):
        """Sets stats from a synced tensor."""
        res = data.tolist()
        self.tp, self.fp, self.fn, self.tn = res[0], res[1], res[2], res[3]

    def sync(self):
        """Deprecated: Use MetricsTracker.sync() for batch synchronization."""
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        data = self.get_stats_tensor(device="cuda")
        torch.distributed.all_reduce(data, op=torch.distributed.ReduceOp.SUM)
        self.set_stats_from_tensor(data)


class RoofTypeMetrics:
    """Multi-class accuracy for roof type classification."""

    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        self.correct = 0.0
        self.total = 0.0
        self.per_class_correct = np.zeros(self.num_classes)
        self.per_class_total = np.zeros(self.num_classes)
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0
        self.per_class_correct = np.zeros(self.num_classes)
        self.per_class_total = np.zeros(self.num_classes)

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            logits: (B, C, H, W)
            targets: (B, H, W) class indices
        """
        with torch.no_grad():
            preds = logits.argmax(dim=1)  # (B, H, W)
            if targets.ndim == 4:
                targets = targets.squeeze(1)
            # Only evaluate where target is not background (0)
            mask = targets > 0
            if mask.sum() > 0:
                self.correct += (preds[mask] == targets[mask]).sum().item()
                self.total += mask.sum().item()
                for c in range(self.num_classes):
                    c_mask = targets == c
                    if c_mask.sum() > 0:
                        self.per_class_correct[c] += (preds[c_mask] == c).sum().item()
                        self.per_class_total[c] += c_mask.sum().item()

    @property
    def accuracy(self) -> float:
        return self.correct / (self.total + 1e-8)

    def compute(self) -> Dict[str, float]:
        result = {"roof_type_accuracy": float(self.accuracy)}
        classes = ["Background", "RCC", "Tiled", "Tin", "Others"]
        for c in range(self.num_classes):
            acc = self.per_class_correct[c] / (self.per_class_total[c] + 1e-8)
            result[f"roof_{classes[c]}_acc"] = float(acc)
        return result

    def get_stats_tensor(self, device: str = "cuda") -> torch.Tensor:
        """Returns stats as a tensor for consolidated sync."""
        # Stats: [correct, total, per_class_correct..., per_class_total...]
        stats: List[float] = [float(self.correct), float(self.total)]
        stats.extend(self.per_class_correct.tolist())
        stats.extend(self.per_class_total.tolist())
        return torch.tensor(stats, dtype=torch.float64, device=device)

    def set_stats_from_tensor(self, data: torch.Tensor):
        """Sets stats from a synced tensor."""
        res = data.tolist()
        self.correct = res[0]
        self.total = res[1]
        offset = 2
        self.per_class_correct = np.array(res[offset : offset + self.num_classes])
        offset += self.num_classes
        self.per_class_total = np.array(res[offset : offset + self.num_classes])

    def sync(self):
        """Deprecated: Use MetricsTracker.sync() for batch synchronization."""
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        data = self.get_stats_tensor(device="cuda")
        torch.distributed.all_reduce(data, op=torch.distributed.ReduceOp.SUM)
        self.set_stats_from_tensor(data)


class MetricsTracker:
    """Tracks all task metrics across an epoch."""

    BINARY_TASKS = [
        "building",
        "road",
        "road_centerline",
        "waterbody",
        "waterbody_line",
        "utility_line",
    ]

    def __init__(self, threshold: float = 0.5, num_roof_classes: int = 5):
        self.binary_metrics = {
            task: TaskMetrics(task, threshold) for task in self.BINARY_TASKS
        }
        self.roof_metrics = RoofTypeMetrics(num_roof_classes)

    def reset(self):
        for m in self.binary_metrics.values():
            m.reset()
        self.roof_metrics.reset()

    def update(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ):
        valid_mask = targets.get("valid_mask", None)

        for task in self.BINARY_TASKS:
            key = f"{task}_mask"
            if key in predictions and key in targets:
                self.binary_metrics[task].update(
                    predictions[key], targets[key], mask=valid_mask
                )

        if "roof_type_mask" in predictions and "roof_type_mask" in targets:
            # RoofTypeMetrics already has an internal mask check for targets > 0
            # which is essentially "onlyPixelsUnderShapefile".
            # We add valid_mask to ensure NoData is also excluded.
            # However, since RoofType target is 0 for background anyway,
            # Targets > 0 already excludes background and NoData.
            self.roof_metrics.update(
                predictions["roof_type_mask"], targets["roof_type_mask"]
            )

    def compute(self) -> Dict[str, float]:
        result = {}
        ious = []
        dices = []

        for task, m in self.binary_metrics.items():
            metrics = m.compute()
            result.update(metrics)
            ious.append(m.iou)
            dices.append(m.dice)

        result.update(self.roof_metrics.compute())

        # Aggregates
        result["avg_iou"] = float(np.mean(ious)) if ious else 0.0
        result["avg_dice"] = float(np.mean(dices)) if dices else 0.0

        return result

    def sync(self):
        """Synchronize all internal metrics across processes using a single all_reduce."""
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return

        # 1. Collect all tensors
        tensors = []
        for task in self.BINARY_TASKS:
            tensors.append(self.binary_metrics[task].get_stats_tensor("cuda"))
        tensors.append(self.roof_metrics.get_stats_tensor("cuda"))
        
        # 2. Concatenate into one large vector
        combined = torch.cat(tensors)
        
        # 3. Single all_reduce
        torch.distributed.all_reduce(combined, op=torch.distributed.ReduceOp.SUM)
        
        # 4. Redistribute values
        offset = 0
        for task in self.BINARY_TASKS:
            chunk = combined[offset : offset + 4]
            self.binary_metrics[task].set_stats_from_tensor(chunk)
            offset += 4
            
        roof_chunk_size = 2 + (2 * self.roof_metrics.num_classes)
        self.roof_metrics.set_stats_from_tensor(combined[offset : offset + roof_chunk_size])
