from __future__ import annotations

"""
Task 3 model: a mine predictor that can "think longer".

I reuse the Task 1 setup (visible board -> mine logits), but I add an explicit sequential
reasoning loop:

- I encode the board into H*W tokens.
- Then I apply the same Transformer block repeatedly (shared weights).
- After each iteration, I can produce mine logits.

If I run more iterations, the model gets more computation budget to refine predictions.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


@dataclass(frozen=True)
class ThinkingMinePredictorConfig:
    vocab_size: int = 11
    d_model: int = 96
    conv_blocks: int = 4
    n_heads: int = 6
    mlp_ratio: int = 4
    dropout: float = 0.1
    height: int = 22
    width: int = 22
    # This is how many iterations I run by default at inference.
    default_steps: int = 4


class ThinkingMinePredictor(nn.Module):
    """
    Mine prediction with an explicit "thinking steps" parameter.

    forward(..., steps=K) runs K repeated reasoning iterations and returns logits.
    If return_all=True, I also return the per-step logits so I can plot how predictions evolve.
    """

    def __init__(self, cfg: ThinkingMinePredictorConfig):
        super().__init__()
        self.cfg = cfg

        d = int(cfg.d_model)
        in_ch = int(cfg.vocab_size)

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, d, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=d),
            nn.SiLU(),
        )
        self.conv = nn.Sequential(*[ResidualConvBlock(d) for _ in range(int(cfg.conv_blocks))])

        self.row_pos = nn.Parameter(torch.zeros(cfg.height, d))
        self.col_pos = nn.Parameter(torch.zeros(cfg.width, d))

        # One shared "thinking" block (weights reused each iteration).
        self.think_block = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=int(cfg.n_heads),
            dim_feedforward=int(cfg.mlp_ratio) * d,
            dropout=float(cfg.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.head = nn.Conv2d(d, 1, kernel_size=1)
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.row_pos, mean=0.0, std=0.02)
        nn.init.normal_(self.col_pos, mean=0.0, std=0.02)

    @staticmethod
    def _to_vocab_idx(x_int8: torch.Tensor) -> torch.Tensor:
        return (x_int8.to(torch.int64) + 1).clamp_(0, 10)

    def _encode_tokens(self, x_int8: torch.Tensor) -> torch.Tensor:
        b, h, w = x_int8.shape
        if h != self.cfg.height or w != self.cfg.width:
            raise ValueError(f"Expected board {self.cfg.height}x{self.cfg.width}, got {h}x{w}")

        idx = self._to_vocab_idx(x_int8)
        x_oh = F.one_hot(idx, num_classes=self.cfg.vocab_size).float()  # (B,H,W,C)
        x_oh = x_oh.permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)

        feats = self.conv(self.stem(x_oh))  # (B,d,H,W)
        tokens = feats.permute(0, 2, 3, 1).reshape(b, h * w, self.cfg.d_model)  # (B,HW,d)

        pos = (self.row_pos[:, None, :] + self.col_pos[None, :, :]).reshape(h * w, self.cfg.d_model)
        return tokens + pos[None, :, :]

    def forward(
        self,
        x_int8: torch.Tensor,
        *,
        steps: Optional[int] = None,
        return_all: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x_int8: (B,H,W) values in [-1..9]
            steps: how many "thinking" iterations to run
            return_all: if True, return (final_logits, logits_per_step)

        Returns:
            final_logits: (B,H,W)
            logits_per_step: list of (B,H,W) if return_all=True
        """
        steps_i = int(self.cfg.default_steps if steps is None else steps)
        steps_i = max(1, steps_i)

        b, h, w = x_int8.shape
        tokens = self._encode_tokens(x_int8)  # (B,HW,d)

        per_step: List[torch.Tensor] = []
        for _ in range(steps_i):
            tokens = self.think_block(tokens)  # (B,HW,d)
            out = tokens.reshape(b, h, w, self.cfg.d_model).permute(0, 3, 1, 2).contiguous()
            logits = self.head(out).squeeze(1)  # (B,H,W)
            if return_all:
                per_step.append(logits)

        if return_all:
            return logits, per_step
        return logits


