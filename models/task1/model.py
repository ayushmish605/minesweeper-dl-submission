from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
class MinePredictorConfig:
    # Visible encoding has 11 distinct values: -1..9
    vocab_size: int = 11
    d_model: int = 96
    conv_blocks: int = 4
    transformer_layers: int = 4
    n_heads: int = 6
    mlp_ratio: int = 4
    dropout: float = 0.1
    height: int = 22
    width: int = 22


class MinePredictor(nn.Module):
    """
    For Task 1, I want a model that takes the *visible* board and outputs a mine
    probability for every cell.

    I start with a small conv/residual stack to pick up local patterns (classic Minesweeper
    motifs near clue numbers), and then I add a Transformer over all (H × W) cells so
    the model can share information globally instead of being trapped in a local CNN receptive
    field. Finally, a 1×1 head turns features into per-cell mine logits.
    """

    def __init__(self, cfg: MinePredictorConfig):
        super().__init__()
        self.cfg = cfg

        in_ch = int(cfg.vocab_size)
        d = int(cfg.d_model)

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, d, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=d),
            nn.SiLU(),
        )
        self.conv = nn.Sequential(*[ResidualConvBlock(d) for _ in range(int(cfg.conv_blocks))])

        # Learned 2D positional embeddings (row+col)
        self.row_pos = nn.Parameter(torch.zeros(cfg.height, d))
        self.col_pos = nn.Parameter(torch.zeros(cfg.width, d))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=int(cfg.n_heads),
            dim_feedforward=int(cfg.mlp_ratio) * d,
            dropout=float(cfg.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=int(cfg.transformer_layers))

        self.head = nn.Conv2d(d, 1, kernel_size=1)

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.row_pos, mean=0.0, std=0.02)
        nn.init.normal_(self.col_pos, mean=0.0, std=0.02)

    @staticmethod
    def _to_vocab_idx(x_int8: torch.Tensor) -> torch.Tensor:
        """
        Map int8 grid values in [-1..9] to indices [0..10].
        """
        return (x_int8.to(torch.int64) + 1).clamp_(0, 10)

    def forward(self, x_int8: torch.Tensor, *, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x_int8: (B,H,W) int8/int64 tensor with values in [-1..9]
            key_padding_mask: optional (B, HW) bool mask for transformer (True = ignore token)

        Returns:
            mine_logits: (B,H,W) float tensor
        """
        b, h, w = x_int8.shape
        if h != self.cfg.height or w != self.cfg.width:
            raise ValueError(f"Expected board {self.cfg.height}x{self.cfg.width}, got {h}x{w}")

        idx = self._to_vocab_idx(x_int8)
        x_oh = F.one_hot(idx, num_classes=self.cfg.vocab_size).float()  # (B,H,W,C)
        x_oh = x_oh.permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)

        feats = self.conv(self.stem(x_oh))  # (B,d,H,W)
        tokens = feats.permute(0, 2, 3, 1).reshape(b, h * w, self.cfg.d_model)  # (B,HW,d)

        # Add 2D pos embedding
        pos = (self.row_pos[:, None, :] + self.col_pos[None, :, :]).reshape(h * w, self.cfg.d_model)
        tokens = tokens + pos[None, :, :]

        tokens = self.transformer(tokens, src_key_padding_mask=key_padding_mask)  # (B,HW,d)
        out = tokens.reshape(b, h, w, self.cfg.d_model).permute(0, 3, 1, 2).contiguous()  # (B,d,H,W)

        mine_logits = self.head(out).squeeze(1)  # (B,H,W)
        return mine_logits


