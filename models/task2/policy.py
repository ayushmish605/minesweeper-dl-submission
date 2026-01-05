from __future__ import annotations

"""
Task 2 policy helpers.

For Task 2 I treat the model's value head as predicting a *survival score* in [0,1] for a
candidate click (row, col). In the notebook, survival is defined as:

  survival = (step of first mine trigger) / (total steps to reach 100% progress)

and I choose actions using a combined score:

  score = predicted_survival - mine_penalty * P(mine)
"""

from typing import List, Optional, Tuple

import numpy as np
import torch

from minesweeper.game import MinesweeperGame
from minesweeper.logic_bot import LogicBot
from models.task1.encoding import ENC_UNREVEALED, visible_to_int8
from .model import MoveValuePredictor
from .value_map_model import BoardValuePredictor


@torch.no_grad()
def select_best_unrevealed_by_value(
    model: MoveValuePredictor,
    visible_board: List[List[str]],
    *,
    device: torch.device,
    batch_size: int = 256,
    max_candidates: Optional[int] = None,
    seed: Optional[int] = None,
    use_logic_prefilter: bool = False,
    only_inferred_safe: bool = False,
    allowed_coords: Optional[np.ndarray] = None,
    epsilon: float = 0.0,
    top_k: int = 1,
) -> Optional[Tuple[int, int]]:
    x = visible_to_int8(visible_board)  # (H,W)
    unrevealed = (x == ENC_UNREVEALED)
    if not np.any(unrevealed):
        return None

    # Optional remediation: use simple deterministic inference from visible clues.
    # This is intentionally lightweight (not the full LogicBot), and it never uses the hidden mines.
    inferred_safe: set[tuple[int, int]] = set()
    inferred_mine: set[tuple[int, int]] = set()
    if bool(use_logic_prefilter):
        try:
            h = int(len(visible_board))
            w = int(len(visible_board[0])) if h > 0 else 0

            def _neighbors(r: int, c: int):
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr = r + dr
                        cc = c + dc
                        if 0 <= rr < h and 0 <= cc < w:
                            yield (rr, cc)

            # Start with mines already shown on the visible board.
            for r in range(h):
                row = visible_board[r]
                for c in range(w):
                    v = row[c]
                    if v == "M":
                        inferred_mine.add((r, c))

            changed = True
            # Iterate a few times; board is tiny so this is cheap.
            for _ in range(20):
                if not changed:
                    break
                changed = False
                for r in range(h):
                    for c in range(w):
                        v = visible_board[r][c]
                        if v in ("E", "M", "F"):
                            continue
                        # "B" is blank (0).
                        if v == "B":
                            clue = 0
                        else:
                            try:
                                clue = int(v)
                            except Exception:
                                continue

                        neigh = list(_neighbors(r, c))
                        unknown = [(rr, cc) for (rr, cc) in neigh if visible_board[rr][cc] == "E"]
                        if not unknown:
                            continue
                        mines = sum(1 for (rr, cc) in neigh if (rr, cc) in inferred_mine or visible_board[rr][cc] == "M")

                        # If all mines around this clue are already accounted for, the rest are safe.
                        if mines == clue:
                            for rc in unknown:
                                if rc not in inferred_safe:
                                    inferred_safe.add(rc)
                                    changed = True

                        # If the remaining unknowns must all be mines, mark them.
                        if clue - mines == len(unknown):
                            for rc in unknown:
                                if rc not in inferred_mine:
                                    inferred_mine.add(rc)
                                    changed = True
        except Exception:
            inferred_safe = set()
            inferred_mine = set()

    # Optional speed knob: evaluate only a subset of candidate actions.
    # I prefer "frontier" cells (unrevealed cells adjacent to at least one revealed cell),
    # since those are generally the most informative moves in Minesweeper.
    coords = None
    try:
        mc = int(max_candidates) if max_candidates is not None else None
    except Exception:
        mc = None

    if mc is not None and mc > 0:
        try:
            revealed = ~unrevealed
            rp = np.pad(revealed.astype(np.uint8), 1, mode="constant", constant_values=0)
            neigh = (
                rp[:-2, :-2]
                + rp[:-2, 1:-1]
                + rp[:-2, 2:]
                + rp[1:-1, :-2]
                + rp[1:-1, 2:]
                + rp[2:, :-2]
                + rp[2:, 1:-1]
                + rp[2:, 2:]
            )
            frontier = unrevealed & (neigh > 0)
            if np.any(frontier):
                coords = np.argwhere(frontier)  # (K,2)
        except Exception:
            coords = None

    # Candidate set (possibly prefiltered).
    if coords is None:
        if bool(use_logic_prefilter):
            # If we found guaranteed-safe moves, only score those.
            if inferred_safe:
                coords = np.asarray(sorted(inferred_safe), dtype=np.int64)
            else:
                if bool(only_inferred_safe):
                    return None
                # Otherwise: avoid inferred mines, score the rest.
                if inferred_mine:
                    try:
                        mask = unrevealed.copy()
                        for (rr, cc) in inferred_mine:
                            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]:
                                mask[rr, cc] = False
                        coords = np.argwhere(mask)
                    except Exception:
                        coords = np.argwhere(unrevealed)
                else:
                    coords = np.argwhere(unrevealed)  # (K,2) rows,cols
        else:
            coords = np.argwhere(unrevealed)  # (K,2) rows,cols
    if coords.size == 0:
        return None

    # Optional external filtering (e.g., from LogicBot inference).
    if allowed_coords is not None:
        try:
            ac = np.asarray(allowed_coords, dtype=np.int64).reshape(-1, 2)
            if ac.size == 0:
                return None
            allow = set((int(r), int(c)) for (r, c) in ac.tolist())
            coords = np.asarray([rc for rc in coords.tolist() if (int(rc[0]), int(rc[1])) in allow], dtype=np.int64)
            if coords.size == 0:
                return None
        except Exception:
            pass

    if mc is not None and mc > 0 and int(coords.shape[0]) > mc:
        try:
            rng = np.random.default_rng(int(seed) if seed is not None else None)
            idx = rng.choice(int(coords.shape[0]), size=int(mc), replace=False)
            coords = coords[idx]
        except Exception:
            coords = coords[: int(mc)]

    # Optional exploration: random move among candidates.
    try:
        eps = float(epsilon)
    except Exception:
        eps = 0.0
    if eps > 0:
        rng = np.random.default_rng(int(seed) if seed is not None else None)
        if float(rng.random()) < eps:
            rc = coords[int(rng.integers(0, int(coords.shape[0])))]
            return (int(rc[0]), int(rc[1]))

    # Performance-critical path: encode the board once, then score many candidate actions.
    xt = torch.from_numpy(x).to(device=device).unsqueeze(0)  # (1,H,W)
    tokens, global_feat = model.encode_board(xt)  # (1,HW,d), (1,d)

    coords_t = torch.from_numpy(coords.astype(np.int64)).to(device=device)  # (N,2)
    vals = model.score_actions_from_encoding(tokens=tokens, global_feat=global_feat, action_rc=coords_t)  # (N,)

    # Pick best (or among top-k).
    k = max(1, int(top_k))
    if k == 1 or int(coords_t.shape[0]) == 1:
        j = int(torch.argmax(vals).item())
        return (int(coords_t[j, 0].item()), int(coords_t[j, 1].item()))

    k = min(k, int(coords_t.shape[0]))
    topv, topi = torch.topk(vals, k=k, largest=True)
    # Choose uniformly among top-k (simple exploration that matches the PDF's "always best?" hint).
    try:
        rng = np.random.default_rng(int(seed) if seed is not None else None)
        pick = int(rng.integers(0, k))
    except Exception:
        pick = 0
    j = int(topi[pick].item())
    return (int(coords_t[j, 0].item()), int(coords_t[j, 1].item()))


@torch.no_grad()
def select_best_unrevealed_from_value_map(
    model: BoardValuePredictor,
    visible_board: List[List[str]],
    *,
    device: torch.device,
    allowed_coords: Optional[np.ndarray] = None,
    epsilon: float = 0.0,
    top_k: int = 1,
    seed: Optional[int] = None,
    mine_penalty: float = 4.0,
) -> Optional[Tuple[int, int]]:
    """
    Actor helper for the value-map model: compute values for all cells in one forward pass,
    then choose among unrevealed cells (optionally restricted via allowed_coords).
    """
    x = visible_to_int8(visible_board)  # (H,W)
    unrevealed = (x == ENC_UNREVEALED)
    if not np.any(unrevealed):
        return None

    if allowed_coords is None:
        coords = np.argwhere(unrevealed).astype(np.int64)
    else:
        coords = np.asarray(allowed_coords, dtype=np.int64).reshape(-1, 2)
        if coords.size == 0:
            return None

    # Epsilon random among candidates.
    try:
        eps = float(epsilon)
    except Exception:
        eps = 0.0
    if eps > 0:
        rng = np.random.default_rng(int(seed) if seed is not None else None)
        if float(rng.random()) < eps:
            rc = coords[int(rng.integers(0, int(coords.shape[0])))]
            return (int(rc[0]), int(rc[1]))

    xt = torch.from_numpy(x).to(device=device).unsqueeze(0)  # (1,H,W)
    value_map, mine_logit_map = model(xt.to(torch.int64))  # (1,H,W), (1,H,W)
    # The model is trained on a survival target in [0,1], but the head is linear.
    # For action selection, I bound it to [0,1] so the mine penalty stays meaningful.
    value_map = torch.sigmoid(value_map[0])  # (H,W)
    mine_prob = torch.sigmoid(mine_logit_map[0])  # (H,W)
    try:
        mp = float(mine_penalty)
    except Exception:
        mp = 0.0
    score_map = value_map - (mp * mine_prob)

    # Gather candidate values.
    rr = torch.from_numpy(coords[:, 0]).to(device=device, dtype=torch.int64)
    cc = torch.from_numpy(coords[:, 1]).to(device=device, dtype=torch.int64)
    vals = score_map[rr, cc]  # (N,)

    k = max(1, int(top_k))
    if k == 1 or int(vals.numel()) == 1:
        j = int(torch.argmax(vals).item())
        return (int(coords[j, 0]), int(coords[j, 1]))

    k = min(k, int(vals.numel()))
    topv, topi = torch.topk(vals, k=k, largest=True)
    try:
        rng = np.random.default_rng(int(seed) if seed is not None else None)
        pick = int(rng.integers(0, k))
    except Exception:
        pick = 0
    j = int(topi[pick].item())
    return (int(coords[j, 0]), int(coords[j, 1]))


def logic_infer_sets(bot: LogicBot) -> None:
    """
    Run full LogicBot inference until saturation.
    This is used for masking and identifying provably-safe moves.
    """
    bot._sync_sets_with_visible()
    bot._update_clue_numbers()
    while bot.make_inferences():
        pass


def allowed_coords_from_logic(bot: LogicBot, game: MinesweeperGame) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Return (safe_coords, guess_coords) based on LogicBot inference.

    - safe_coords: inferred-safe unrevealed cells
    - guess_coords: all unrevealed cells excluding inferred mines
    """
    logic_infer_sets(bot)
    safe = [(int(r), int(c)) for (r, c) in bot.inferred_safe if game.board[int(r)][int(c)] == "E"]
    mines = {(int(r), int(c)) for (r, c) in bot.inferred_mine}

    guess: list[tuple[int, int]] = []
    for r in range(int(game.height)):
        row = game.board[r]
        for c in range(int(game.width)):
            if row[c] != "E":
                continue
            if (int(r), int(c)) in mines:
                continue
            guess.append((int(r), int(c)))

    safe_coords = np.asarray(sorted(safe), dtype=np.int64) if safe else None
    guess_coords = np.asarray(guess, dtype=np.int64) if guess else None
    return safe_coords, guess_coords


@torch.no_grad()
def actor_choose_click_value_map(
    *,
    model: BoardValuePredictor,
    game: MinesweeperGame,
    bot: LogicBot,
    device: torch.device,
    seed: int,
    mine_penalty: float = 4.0,
    epsilon: float = 0.0,
    top_k: int = 1,
    use_logic_mask: bool = True,
    use_model_on_safe_moves: bool = True,
) -> Optional[Tuple[int, int]]:
    """
    Task 2 actor (value-map version).

    - If use_logic_mask=True, restrict choices to:
      - inferred-safe cells if any exist, else
      - all unrevealed cells excluding inferred mines.
    - Otherwise, score all unrevealed cells.
    """
    visible = game.get_visible_board()
    allowed = None
    if bool(use_logic_mask):
        safe_coords, guess_coords = allowed_coords_from_logic(bot, game)
        has_safe = (safe_coords is not None and int(safe_coords.shape[0]) > 0)

        # Optional "guess-only" mode: if LogicBot has a provably-safe move, I never override it.
        # This is especially useful on easy difficulty where LogicBot is already near ceiling.
        if bool(has_safe) and (not bool(use_model_on_safe_moves)):
            a = bot.select_action()
            return None if a is None else (int(a[0]), int(a[1]))

        allowed = safe_coords if bool(has_safe) else guess_coords

    return select_best_unrevealed_from_value_map(
        model,
        visible,
        device=device,
        allowed_coords=allowed,
        epsilon=float(epsilon),
        top_k=int(top_k),
        seed=int(seed),
        mine_penalty=float(mine_penalty),
    )

