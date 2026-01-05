"""
I use this script to export a saved game-state JSON into a compact `.npz` for training.

What I assume is present:
- each state has an `action` ([row, col] or null)
- and (in the simplest case) each state has a full `board` snapshot (HxW strings)

I *don't* require per-state diffs in the JSON; I can derive diffs by comparing successive
boards during export.

Usage:
  .venv/bin/python scripts/export_game_state_json_to_npz.py \
    --input data/game_states/game_*.json \
    --output data/npz_exports/

By default, I also rewrite the JSON in-place to remove any lingering `"diff"` keys. You can
disable that with `--no-rewrite-json`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# Board encoding for NN inputs (int8):
#  -1: unrevealed ("E")
#   0: blank revealed ("B" or "0")
#  1..8: clue numbers
#   9: mine shown ("M")  (should generally only appear in terminal/loss snapshots)
ENC_UNREVEALED = -1
ENC_BLANK = 0
ENC_MINE_SHOWN = 9


def encode_cell(v: Any) -> int:
    if v is None:
        return ENC_UNREVEALED
    if isinstance(v, bool):
        return ENC_UNREVEALED
    if isinstance(v, (int, float)):
        # Coerce numeric clues into 0..8
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


def extract_board(state: Dict[str, Any]) -> List[List[Any]]:
    inner = state.get("state", state)
    board = inner.get("board") or inner.get("visible_board") or inner.get("visible") or inner.get("grid") or []
    return board if isinstance(board, list) else []


def extract_action(state: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    inner = state.get("state", state)
    a = inner.get("action") or state.get("action")
    if a is None:
        return None
    if isinstance(a, tuple) and len(a) == 2:
        return int(a[0]), int(a[1])
    if isinstance(a, list) and len(a) == 2:
        return int(a[0]), int(a[1])
    return None


def strip_diff_keys_inplace(data: Dict[str, Any]) -> bool:
    changed = False
    states = data.get("states")
    if isinstance(states, list):
        for s in states:
            if not isinstance(s, dict):
                continue
            if "diff" in s:
                s.pop("diff", None)
                changed = True
            inner = s.get("state")
            if isinstance(inner, dict) and "diff" in inner:
                inner.pop("diff", None)
                changed = True
    return changed


def derive_step_diffs(flat_prev: np.ndarray, flat_now: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (changed_idx, new_values) where:
      - changed_idx: int32 indices into flattened board
      - new_values: int8 encoded new values at those indices
    """
    changed = np.nonzero(flat_prev != flat_now)[0].astype(np.int32)
    new_vals = flat_now[changed].astype(np.int8)
    return changed, new_vals


@dataclass
class ExportResult:
    output_path: Path
    num_steps: int
    height: int
    width: int
    max_changes: int


def export_one(input_path: Path, output_dir: Path, rewrite_json: bool = True) -> ExportResult:
    with open(input_path, "r") as f:
        data: Dict[str, Any] = json.load(f)

    if rewrite_json:
        changed = strip_diff_keys_inplace(data)
        if changed:
            now_iso = datetime.now().isoformat()
            data["last_updated"] = now_iso
            if isinstance(data.get("metadata"), dict):
                data["metadata"]["last_updated"] = now_iso
            with open(input_path, "w") as f:
                json.dump(data, f, indent=2)

    states = data.get("states", [])
    if not isinstance(states, list) or not states:
        raise ValueError(f"No states found in {input_path}")

    # Determine H, W from first non-empty board
    first_board: List[List[Any]] = []
    for s in states:
        if isinstance(s, dict):
            b = extract_board(s)
            if b:
                first_board = b
                break
    if not first_board:
        raise ValueError(f"No board snapshots found in {input_path}")

    H = int((data.get("metadata") or {}).get("height") or len(first_board))
    W = int((data.get("metadata") or {}).get("width") or (len(first_board[0]) if first_board and first_board[0] else 0))
    if H <= 0 or W <= 0:
        raise ValueError(f"Invalid board dimensions H={H}, W={W} in {input_path}")

    N = len(states)

    boards = np.empty((N, H, W), dtype=np.int8)
    actions = np.full((N, 2), -1, dtype=np.int16)
    mine_shown = np.zeros((N,), dtype=np.uint8)

    # First pass: encode boards + actions
    for i, s in enumerate(states):
        if not isinstance(s, dict):
            # empty fallback
            boards[i, :, :] = ENC_UNREVEALED
            continue

        b = extract_board(s)
        a = extract_action(s)
        if a is not None:
            actions[i, 0] = a[0]
            actions[i, 1] = a[1]

        # Encode board, padding missing cells as unrevealed
        mine_flag = 0
        for r in range(H):
            row = b[r] if r < len(b) and isinstance(b[r], list) else []
            for c in range(W):
                v = row[c] if c < len(row) else "E"
                enc = encode_cell(v)
                boards[i, r, c] = enc
                if enc == ENC_MINE_SHOWN:
                    mine_flag = 1
        mine_shown[i] = mine_flag

    # Second pass: derive diffs (flattened)
    flat = boards.reshape(N, H * W)
    idx_list: List[np.ndarray] = []
    val_list: List[np.ndarray] = []
    max_changes = 0

    # implicit previous for i=0 is all unrevealed
    prev0 = np.full((H * W,), ENC_UNREVEALED, dtype=np.int8)
    idx0, val0 = derive_step_diffs(prev0, flat[0])
    idx_list.append(idx0)
    val_list.append(val0)
    max_changes = max(max_changes, int(idx0.shape[0]))

    for i in range(1, N):
        idx, val = derive_step_diffs(flat[i - 1], flat[i])
        idx_list.append(idx)
        val_list.append(val)
        if idx.shape[0] > max_changes:
            max_changes = int(idx.shape[0])

    diff_len = np.zeros((N,), dtype=np.int32)
    diff_idx = np.full((N, max_changes), -1, dtype=np.int32)
    diff_val = np.full((N, max_changes), 0, dtype=np.int8)

    for i in range(N):
        k = int(idx_list[i].shape[0])
        diff_len[i] = k
        if k:
            diff_idx[i, :k] = idx_list[i]
            diff_val[i, :k] = val_list[i]

    meta = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
    meta_out = {
        "source_json": str(input_path),
        "mode": meta.get("mode"),
        "height": H,
        "width": W,
        "num_mines": meta.get("num_mines"),
        "seed": meta.get("seed"),
        "created_at": meta.get("created_at"),
        "last_updated": meta.get("last_updated") or data.get("last_updated"),
        "game_state": meta.get("game_state"),
        "encoding": {
            "E": ENC_UNREVEALED,
            "B_or_0": ENC_BLANK,
            "1_to_8": "1..8",
            "M": ENC_MINE_SHOWN,
        },
    }
    meta_json = json.dumps(meta_out, ensure_ascii=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (input_path.stem + ".npz")
    np.savez_compressed(
        out_path,
        boards=boards,              # int8, [N,H,W]
        actions=actions,            # int16, [N,2]
        mine_shown=mine_shown,      # uint8, [N]
        diff_len=diff_len,          # int32, [N]
        diff_idx=diff_idx,          # int32, [N,max_changes]
        diff_val=diff_val,          # int8,  [N,max_changes]
        meta_json=np.array(meta_json),  # 0-d string array
    )

    return ExportResult(out_path, N, H, W, max_changes)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to game_*.json (or directory)")
    p.add_argument("--output", required=True, help="Output directory for .npz")
    p.add_argument("--no-rewrite-json", action="store_true", help="Do not remove 'diff' keys from JSON in-place")
    args = p.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)

    rewrite = not args.no_rewrite_json

    if in_path.is_dir():
        json_files = sorted(in_path.glob("game_*.json"))
        if not json_files:
            raise SystemExit(f"No game_*.json files found in {in_path}")
        for jf in json_files:
            res = export_one(jf, out_dir, rewrite_json=rewrite)
            print(f"Wrote {res.output_path} (N={res.num_steps}, H={res.height}, W={res.width}, max_changes={res.max_changes})")
        return 0

    res = export_one(in_path, out_dir, rewrite_json=rewrite)
    print(f"Wrote {res.output_path} (N={res.num_steps}, H={res.height}, W={res.width}, max_changes={res.max_changes})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


