"""
evaluation/metrics.py — Segmentation metrics
---------------------------------------------
All metrics operate on numpy arrays or torch tensors.
Used by the Evaluator, weather_analysis, and class_analysis modules.
"""

from typing import Dict, List, Optional

import numpy as np
import torch


class SegmentationMetrics:
    """Accumulates confusion matrix over batches, then computes metrics.

    Usage:
        metrics = SegmentationMetrics(num_classes=19, ignore_index=255)
        for pred, target in dataloader:
            metrics.update(pred, target)
        results = metrics.compute()
        metrics.reset()
    """

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(
        self,
        preds:   "torch.Tensor | np.ndarray",   # (B, H, W) int class indices
        targets: "torch.Tensor | np.ndarray",   # (B, H, W) int class indices
    ):
        if isinstance(preds,   torch.Tensor): preds   = preds.cpu().numpy()
        if isinstance(targets, torch.Tensor): targets = targets.cpu().numpy()

        preds   = preds.astype(np.int64).flatten()
        targets = targets.astype(np.int64).flatten()

        # Remove ignored pixels
        valid = targets != self.ignore_index
        preds   = preds[valid]
        targets = targets[valid]

        # Clamp predictions to valid range
        preds = np.clip(preds, 0, self.num_classes - 1)

        # Fast confusion matrix accumulation
        idx = targets * self.num_classes + preds
        self.confusion += np.bincount(idx, minlength=self.num_classes ** 2) \
                            .reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, float]:
        """Return dict with mIoU, pixel_accuracy, and per-class IoU."""
        cm = self.confusion.astype(np.float64)

        # Per-class IoU: TP / (TP + FP + FN)
        tp  = np.diag(cm)
        fp  = cm.sum(axis=0) - tp   # col sums minus diagonal
        fn  = cm.sum(axis=1) - tp   # row sums minus diagonal
        denom = tp + fp + fn

        iou_per_class = np.where(denom > 0, tp / denom, np.nan)

        # mIoU over classes that have at least one sample
        valid_classes = ~np.isnan(iou_per_class)
        miou = float(np.nanmean(iou_per_class))

        # Pixel accuracy
        total_correct = float(tp.sum())
        total_pixels  = float(cm.sum())
        pixel_acc     = total_correct / total_pixels if total_pixels > 0 else 0.0

        return {
            "mIoU":           miou,
            "pixel_accuracy": pixel_acc,
            "iou_per_class":  iou_per_class.tolist(),
            "valid_classes":  valid_classes.tolist(),
        }

    def get_class_iou(
        self,
        class_indices: Optional[List[int]] = None,
        class_names:   Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Return IoU for a specific subset of classes (e.g. safety-critical)."""
        results = self.compute()
        ious    = results["iou_per_class"]
        indices = class_indices or list(range(self.num_classes))
        names   = class_names or [str(i) for i in indices]
        return {name: ious[idx] for name, idx in zip(names, indices)}


# ------------------------------------------------------------------
# Standalone functions for quick use
# ------------------------------------------------------------------

def compute_miou(
    preds:       np.ndarray,
    targets:     np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> float:
    """One-shot mIoU computation without accumulation."""
    m = SegmentationMetrics(num_classes, ignore_index)
    m.update(preds, targets)
    return m.compute()["mIoU"]


def compute_robustness_drop(
    clear_miou:   float,
    adverse_miou: float,
) -> Dict[str, float]:
    """Compute absolute and relative performance degradation."""
    drop     = clear_miou - adverse_miou
    rel_drop = (drop / clear_miou * 100) if clear_miou > 0 else 0.0
    return {
        "clear_miou":   clear_miou,
        "adverse_miou": adverse_miou,
        "absolute_drop": drop,
        "relative_drop_pct": rel_drop,
    }
