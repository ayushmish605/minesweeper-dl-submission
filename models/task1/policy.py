from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from .encoding import ENC_UNREVEALED, visible_to_int8
from .model import MinePredictor


@torch.no_grad()
def select_safest_unrevealed(
    model: MinePredictor,
    visible_board: List[List[str]],
    *,
    device: torch.device,
    temperature: float = 1.0,
) -> Optional[Tuple[int, int]]:
    """
    Pick the cell with the lowest predicted mine probability among unrevealed cells.

    temperature > 1.0 flattens probabilities (more exploratory)
    temperature < 1.0 sharpens probabilities (more greedy)
    """
    x = visible_to_int8(visible_board)  # (H,W) int8
    unrevealed = (x == ENC_UNREVEALED)
    if not np.any(unrevealed):
        return None

    xt = torch.from_numpy(x).to(device=device).unsqueeze(0)  # (1,H,W)
    logits = model(xt).squeeze(0)  # (H,W)
    if temperature and float(temperature) != 1.0:
        logits = logits / float(temperature)
    probs = torch.sigmoid(logits)  # mine probability

    # Mask revealed cells to +inf so argmin ignores them
    mask = torch.from_numpy(unrevealed).to(device=device)
    probs_masked = torch.where(mask, probs, torch.full_like(probs, float("inf")))

    flat_idx = torch.argmin(probs_masked.reshape(-1)).item()
    h, w = x.shape
    r = int(flat_idx // w)
    c = int(flat_idx % w)
    return (r, c)


