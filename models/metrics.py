"""
Shared metrics helpers for my Minesweeper DL project.

I keep these here so I don't re-implement the same evaluation logic across notebooks
and tasks (and so I don't accidentally compare apples-to-oranges).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    BCEWithLogits on a masked subset of cells.

    - logits/targets/mask should be broadcastable to the same shape
    - mask should be 0/1 (or boolean); values outside the mask are ignored
    """
    loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    loss = loss * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def pos_weight_from_targets(targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute a scalar pos_weight = neg/pos over the masked elements.
    This is a standard way to counter class imbalance for BCE.
    """
    m = mask.float()
    y = targets.float()
    pos = (y * m).sum()
    neg = ((1.0 - y) * m).sum()
    w = neg / pos.clamp_min(1.0)
    # BCEWithLogits expects a tensor on the same device/dtype.
    return w.to(device=targets.device, dtype=targets.dtype)


@torch.no_grad()
def masked_binary_confusion_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> Dict[str, int]:
    """
    Confusion counts for a masked binary classification problem.
    Returns dict with keys: tp, fp, tn, fn, n.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= float(threshold)).to(dtype=targets.dtype)

    m = mask.float()
    y = targets.float()
    p = preds.float()

    tp = ((p == 1.0) & (y == 1.0)).float() * m
    fp = ((p == 1.0) & (y == 0.0)).float() * m
    tn = ((p == 0.0) & (y == 0.0)).float() * m
    fn = ((p == 0.0) & (y == 1.0)).float() * m

    out = {
        "tp": int(tp.sum().item()),
        "fp": int(fp.sum().item()),
        "tn": int(tn.sum().item()),
        "fn": int(fn.sum().item()),
        "n": int(m.sum().item()),
    }
    return out


def binary_metrics_from_confusion(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    """
    Standard metrics for the positive class:
    - precision/recall/F1 for y==1
    - accuracy overall
    """
    tp = float(tp)
    fp = float(fp)
    tn = float(tn)
    fn = float(fn)

    prec = tp / max(1.0, (tp + fp))
    rec = tp / max(1.0, (tp + fn))
    f1 = (2.0 * prec * rec) / max(1e-12, (prec + rec))
    acc = (tp + tn) / max(1.0, (tp + tn + fp + fn))
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1), "acc": float(acc)}


@torch.no_grad()
def masked_binary_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> Dict[str, float]:
    c = masked_binary_confusion_from_logits(logits, targets, mask, threshold=threshold)
    m = binary_metrics_from_confusion(c["tp"], c["fp"], c["tn"], c["fn"])
    return {**m, **{k: float(c[k]) for k in ("tp", "fp", "tn", "fn", "n")}}


@torch.no_grad()
def regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Simple regression metrics (for Task 2).
    Returns: mse, rmse, mae, corr (Pearson).
    """
    pred = pred.float().view(-1)
    target = target.float().view(-1)
    diff = pred - target
    mse = torch.mean(diff * diff)
    rmse = torch.sqrt(mse.clamp_min(0.0))
    mae = torch.mean(torch.abs(diff))

    # Pearson correlation (guard against zero variance).
    vx = pred - pred.mean()
    vy = target - target.mean()
    denom = (vx.pow(2).mean().sqrt() * vy.pow(2).mean().sqrt()).clamp_min(1e-12)
    corr = (vx * vy).mean() / denom

    return {"mse": float(mse.item()), "rmse": float(rmse.item()), "mae": float(mae.item()), "corr": float(corr.item())}


