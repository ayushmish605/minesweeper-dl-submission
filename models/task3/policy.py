from __future__ import annotations

"""
Task 3 policy: same as Task 1 (click the safest unrevealed cell),
but I can choose how many "thinking" steps to run.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch

from models.task1.encoding import ENC_UNREVEALED, visible_to_int8
from .model import ThinkingMinePredictor


@torch.no_grad()
def select_safest_unrevealed_thinking(
    model: ThinkingMinePredictor,
    visible_board: List[List[str]],
    *,
    device: torch.device,
    steps: int = 4,
    temperature: float = 1.0,
) -> Optional[Tuple[int, int]]:
    x = visible_to_int8(visible_board)
    unrevealed = (x == ENC_UNREVEALED)
    if not np.any(unrevealed):
        return None

    xt = torch.from_numpy(x).to(device=device).unsqueeze(0)  # (1,H,W)
    logits = model(xt, steps=int(steps)).squeeze(0)  # (H,W)
    if temperature and float(temperature) != 1.0:
        logits = logits / float(temperature)
    probs = torch.sigmoid(logits)

    mask = torch.from_numpy(unrevealed).to(device=device)
    probs_masked = torch.where(mask, probs, torch.full_like(probs, float("inf")))
    flat_idx = torch.argmin(probs_masked.reshape(-1)).item()
    h, w = x.shape
    return (int(flat_idx // w), int(flat_idx % w))


