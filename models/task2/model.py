from __future__ import annotations

"""
Task 2 model: predict how "good" a candidate click is.

I keep the structure similar to Task 1:
- local patterns via conv/residual blocks
- global reasoning via a Transformer over H*W tokens

The difference is the head:
- Task 1 head outputs per-cell mine logits
- Task 2 head outputs a single scalar value for a chosen action cell
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
class MoveValuePredictorConfig:
    vocab_size: int = 11  # visible encoding is -1..9 => 11 buckets
    d_model: int = 96
    conv_blocks: int = 4
    transformer_layers: int = 4
    n_heads: int = 6
    mlp_ratio: int = 4
    dropout: float = 0.1
    height: int = 22
    width: int = 22
    value_hidden: int = 128


class MoveValuePredictor(nn.Module):
    """
    Given a visible board and a candidate click (r,c), I predict a scalar "value".

    In my dataset, the value is: how many steps the LogicBot survives after taking that click.
    """

    def __init__(self, cfg: MoveValuePredictorConfig):
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

        vh = int(cfg.value_hidden)
        self.value_head = nn.Sequential(
            nn.Linear(2 * d, vh),
            nn.SiLU(),
            nn.Linear(vh, 1),
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.row_pos, mean=0.0, std=0.02)
        nn.init.normal_(self.col_pos, mean=0.0, std=0.02)

    @staticmethod
    def _to_vocab_idx(x_int8: torch.Tensor) -> torch.Tensor:
        return (x_int8.to(torch.int64) + 1).clamp_(0, 10)

    def _encode_tokens(self, x_int8: torch.Tensor) -> torch.Tensor:
        """
        Return transformer tokens (B, HW, d).
        """
        b, h, w = x_int8.shape
        if h != self.cfg.height or w != self.cfg.width:
            raise ValueError(f"Expected board {self.cfg.height}x{self.cfg.width}, got {h}x{w}")

        idx = self._to_vocab_idx(x_int8)
        x_oh = F.one_hot(idx, num_classes=self.cfg.vocab_size).float()  # (B,H,W,C)
        x_oh = x_oh.permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)

        feats = self.conv(self.stem(x_oh))  # (B,d,H,W)
        tokens = feats.permute(0, 2, 3, 1).reshape(b, h * w, self.cfg.d_model)  # (B,HW,d)

        pos = (self.row_pos[:, None, :] + self.col_pos[None, :, :]).reshape(h * w, self.cfg.d_model)
        tokens = tokens + pos[None, :, :]

        return self.transformer(tokens)  # (B,HW,d)

    def encode_board(self, x_int8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of boards once.

        Returns:
            tokens: (B, HW, d)
            global_feat: (B, d)

        Notes:
            This is primarily a speed hook for acting: when scoring many candidate actions
            on the *same* board, we should not recompute the transformer for each action.
        """
        tokens = self._encode_tokens(x_int8)
        return tokens, tokens.mean(dim=1)

    def score_actions_from_encoding(
        self,
        *,
        tokens: torch.Tensor,
        global_feat: torch.Tensor,
        action_rc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score actions given a precomputed board encoding.

        Args:
            tokens: (B0, HW, d) board tokens
            global_feat: (B0, d) global feature for each encoded board
            action_rc: (B,2) row/col actions

        Returns:
            values: (B,) float

        Supports:
            - scoring many actions for a single board: tokens/global_feat with B0==1
            - scoring aligned batches: tokens/global_feat with B0==B
        """
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be (B0,HW,d), got {tuple(tokens.shape)}")
        if global_feat.ndim != 2:
            raise ValueError(f"global_feat must be (B0,d), got {tuple(global_feat.shape)}")
        if action_rc.ndim != 2 or int(action_rc.shape[1]) != 2:
            raise ValueError(f"action_rc must be (B,2), got {tuple(action_rc.shape)}")

        b0, hw, d = tokens.shape
        b = int(action_rc.shape[0])
        if int(global_feat.shape[0]) != int(b0) or int(global_feat.shape[1]) != int(d):
            raise ValueError(
                f"global_feat must match tokens batch/channels: tokens={tuple(tokens.shape)} global_feat={tuple(global_feat.shape)}"
            )

        # Resolve H/W from cfg (tokens are HW-flattened in row-major order).
        h = int(self.cfg.height)
        w = int(self.cfg.width)
        if int(hw) != int(h * w):
            raise ValueError(f"tokens second dim must be HW={h*w}, got {hw}")

        # Expand encoding if we are scoring many actions for one board.
        if b0 == 1 and b > 1:
            tokens_use = tokens.expand(b, -1, -1)
            global_use = global_feat.expand(b, -1)
        elif b0 == b:
            tokens_use = tokens
            global_use = global_feat
        else:
            raise ValueError(f"Unsupported batch shapes: tokens batch={b0} actions batch={b}")

        r = action_rc[:, 0].clamp(0, h - 1)
        c = action_rc[:, 1].clamp(0, w - 1)
        flat = (r * w + c).to(torch.int64)  # (B,)

        action_feat = tokens_use[torch.arange(b, device=tokens_use.device), flat]  # (B,d)
        feat = torch.cat([global_use, action_feat], dim=-1)  # (B,2d)
        return self.value_head(feat).squeeze(-1)

    def forward(self, x_int8: torch.Tensor, action_rc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_int8: (B,H,W) with values in [-1..9]
            action_rc: (B,2) int64 (row,col)

        Returns:
            values: (B,) float
        """
        tokens, global_feat = self.encode_board(x_int8)
        return self.score_actions_from_encoding(tokens=tokens, global_feat=global_feat, action_rc=action_rc)


