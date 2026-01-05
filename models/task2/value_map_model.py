from __future__ import annotations

"""
Task 2 (improved): predict values for *every* possible move in one forward pass.

This implements two heads:
- value_map: predicts a survival score for clicking each cell (defined in the notebook), with target in [0,1]
- mine_logit_map: predicts P(mine | click this cell) (auxiliary, stabilizes learning)

The actor can then:
- compute value_map once per visible state
- mask illegal cells (revealed) and optional inferred-mine cells
- pick argmax (or sample among top-k / epsilon exploration)
"""

from dataclasses import dataclass

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
class BoardValuePredictorConfig:
    vocab_size: int = 11  # visible encoding is -1..9 => 11 buckets
    d_model: int = 96
    conv_blocks: int = 4
    transformer_layers: int = 4
    n_heads: int = 6
    mlp_ratio: int = 4
    dropout: float = 0.1
    height: int = 22
    width: int = 22
    head_hidden: int = 128


class BoardValuePredictor(nn.Module):
    """
    Input: visible board x_int8 (B,H,W) in [-1..9]

    Output:
      - value_map: (B,H,W) *raw* value head output (trained against a survival target in [0,1])
      - mine_logit_map: (B,H,W) logit for P(mine if clicked)
    """

    def __init__(self, cfg: BoardValuePredictorConfig):
        super().__init__()
        self.cfg = cfg
        d = int(cfg.d_model)

        self.stem = nn.Sequential(
            nn.Conv2d(int(cfg.vocab_size), d, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=d),
            nn.SiLU(),
        )
        self.conv = nn.Sequential(*[ResidualConvBlock(d) for _ in range(int(cfg.conv_blocks))])

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

        hh = int(cfg.head_hidden)
        # Per-cell value uses token + global context.
        self.value_head = nn.Sequential(
            nn.Linear(2 * d, hh),
            nn.SiLU(),
            nn.Linear(hh, 1),
        )
        # Per-cell mine probability can be predicted from the token itself.
        self.mine_head = nn.Sequential(
            nn.Linear(d, hh),
            nn.SiLU(),
            nn.Linear(hh, 1),
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.row_pos, mean=0.0, std=0.02)
        nn.init.normal_(self.col_pos, mean=0.0, std=0.02)

    @staticmethod
    def _to_vocab_idx(x_int8: torch.Tensor) -> torch.Tensor:
        return (x_int8.to(torch.int64) + 1).clamp_(0, 10)

    def encode_tokens(self, x_int8: torch.Tensor) -> torch.Tensor:
        """
        Return transformer tokens (B, HW, d).
        """
        b, h, w = x_int8.shape
        if h != self.cfg.height or w != self.cfg.width:
            raise ValueError(f"Expected board {self.cfg.height}x{self.cfg.width}, got {h}x{w}")

        idx = self._to_vocab_idx(x_int8)
        x_oh = F.one_hot(idx, num_classes=int(self.cfg.vocab_size)).float()  # (B,H,W,C)
        x_oh = x_oh.permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)

        feats = self.conv(self.stem(x_oh))  # (B,d,H,W)
        tokens = feats.permute(0, 2, 3, 1).reshape(b, h * w, int(self.cfg.d_model))  # (B,HW,d)

        pos = (self.row_pos[:, None, :] + self.col_pos[None, :, :]).reshape(h * w, int(self.cfg.d_model))
        tokens = tokens + pos[None, :, :]

        return self.transformer(tokens)  # (B,HW,d)

    def forward(self, x_int8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          value_map: (B,H,W)
          mine_logit_map: (B,H,W)
        """
        b, h, w = x_int8.shape
        tokens = self.encode_tokens(x_int8)  # (B,HW,d)
        global_feat = tokens.mean(dim=1, keepdim=True)  # (B,1,d)

        tok = tokens  # (B,HW,d)
        g = global_feat.expand(-1, tok.shape[1], -1)  # (B,HW,d)
        feat = torch.cat([tok, g], dim=-1)  # (B,HW,2d)

        value = self.value_head(feat).squeeze(-1)  # (B,HW)
        mine_logit = self.mine_head(tok).squeeze(-1)  # (B,HW)

        return value.view(b, h, w), mine_logit.view(b, h, w)

    @torch.no_grad()
    def predict_survival_and_mine_prob_maps(self, x_int8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience for inference/action selection.

        Returns:
          - survival_map: (B,H,W) in [0,1]
          - mine_prob_map: (B,H,W) in [0,1]
        """
        v, mine_logit = self(x_int8)
        return torch.sigmoid(v), torch.sigmoid(mine_logit)


