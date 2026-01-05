from __future__ import annotations

from typing import Any, List

import numpy as np

# Visible-board encoding for NN inputs (int8):
#  -1: unrevealed ("E")
#   0: blank revealed ("B" or "0")
#  1..8: clue numbers
#   9: mine shown ("M")  (may appear if allow_mine_triggers=True)
ENC_UNREVEALED: int = -1
ENC_BLANK: int = 0
ENC_MINE_SHOWN: int = 9


def encode_cell(v: Any) -> int:
    if v is None:
        return ENC_UNREVEALED
    if isinstance(v, (int, np.integer)):
        x = int(v)
        if x <= 0:
            return ENC_BLANK
        return min(8, x)
    s = str(v)
    if s == "E":
        return ENC_UNREVEALED
    if s in {"B", "0"}:
        return ENC_BLANK
    if s == "M":
        return ENC_MINE_SHOWN
    try:
        x = int(s)
        if x <= 0:
            return ENC_BLANK
        return min(8, x)
    except Exception:
        return ENC_UNREVEALED


def visible_to_int8(visible_board: List[List[Any]]) -> np.ndarray:
    """
    Convert a visible board (list[list[str]]) into an int8 array [H,W].
    """
    h = len(visible_board)
    w = len(visible_board[0]) if h and visible_board[0] else 0
    out = np.full((h, w), ENC_UNREVEALED, dtype=np.int8)
    for r in range(h):
        row = visible_board[r] if r < len(visible_board) else []
        for c in range(w):
            v = row[c] if c < len(row) else "E"
            out[r, c] = np.int8(encode_cell(v))
    return out


def mine_mask_from_actual(actual_board: List[List[str]]) -> np.ndarray:
    """
    Return uint8 mask [H,W] where 1 indicates a mine in the actual board.
    """
    h = len(actual_board)
    w = len(actual_board[0]) if h and actual_board[0] else 0
    out = np.zeros((h, w), dtype=np.uint8)
    for r in range(h):
        row = actual_board[r] if r < len(actual_board) else []
        for c in range(w):
            out[r, c] = 1 if (c < len(row) and row[c] == "M") else 0
    return out


def unrevealed_mask_from_visible(visible_board: List[List[str]]) -> np.ndarray:
    """
    Return uint8 mask [H,W] where 1 indicates currently unrevealed cells.
    """
    h = len(visible_board)
    w = len(visible_board[0]) if h and visible_board[0] else 0
    out = np.zeros((h, w), dtype=np.uint8)
    for r in range(h):
        row = visible_board[r] if r < len(visible_board) else []
        for c in range(w):
            out[r, c] = 1 if (c < len(row) and row[c] == "E") else 0
    return out


