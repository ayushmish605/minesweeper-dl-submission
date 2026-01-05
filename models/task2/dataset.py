from __future__ import annotations

"""
Task 2 data generation.

The PDF asks me to predict "how well a bot is going to do" given:
- the current visible board state
- a selected, unrevealed cell to click

So for each recorded state, I sample a handful of candidate clicks, simulate a rollout
starting with that click, and store the resulting survival stats as supervision.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from minesweeper.game import GameState, MinesweeperGame
from minesweeper.logic_bot import LogicBot
from models.task1.encoding import visible_to_int8


@dataclass(frozen=True)
class Task2NPZ:
    x_visible: np.ndarray  # int8, (N,H,W) values in [-1..9]
    action_rc: np.ndarray  # int16, (N,2) (row,col)
    y_steps: np.ndarray  # float32, (N,) predicted "steps survived" (including the first click)
    y_cells_opened: np.ndarray  # float32, (N,) cells opened by the rollout policy
    y_mines_triggered: np.ndarray  # float32, (N,) mines triggered during rollout
    y_game_won: np.ndarray  # uint8, (N,) {0,1}
    meta_json: str


def _unrevealed_cells(game: MinesweeperGame) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for r in range(int(game.height)):
        for c in range(int(game.width)):
            if game.board[r][c] == "E":
                out.append((r, c))
    return out


def _rollout_with_logic_bot(
    game: MinesweeperGame, *, max_steps: int = 512, allow_mine_triggers: bool = False, seed: int = 0
) -> Dict[str, float]:
    """
    Roll the game forward using LogicBot until the game ends (or step cap).

    I return simple scalar stats that are easy to train a network on.
    """
    bot = LogicBot(game, seed=int(seed))
    steps = 0
    while game.get_game_state() == GameState.PROG and steps < int(max_steps):
        steps += 1
        result, _ = bot.play_step()
        if result == "Lost" and not bool(allow_mine_triggers):
            break
        if result in {"Win", "Done"}:
            break

    stats = game.get_statistics()
    return {
        "steps": float(steps),
        "cells_opened": float(stats.get("cells_opened", 0) or 0),
        "mines_triggered": float(stats.get("mines_triggered", 0) or 0),
        "game_won": float(1.0 if stats.get("game_won") else 0.0),
    }


def _simulate_action_then_rollout(
    game: MinesweeperGame,
    *,
    action: Tuple[int, int],
    rollout_max_steps: int,
    allow_mine_triggers: bool,
    seed: int,
    rollouts_per_action: int = 1,
) -> Dict[str, float]:
    """
    Clone the game, apply one candidate click, then let LogicBot play to termination.
    """
    k = max(1, int(rollouts_per_action))
    outs: List[Dict[str, float]] = []

    r, c = int(action[0]), int(action[1])
    for i in range(k):
        # `copy.deepcopy(game)` is noticeably slow and dominates dataset generation.
        # I only need a faithful copy of the game state (boards + a few scalars), so I do a
        # lightweight manual clone here.
        g2: MinesweeperGame = _clone_game_fast(game)
        setattr(g2, "allow_mine_triggers", bool(allow_mine_triggers))

        _ = g2.player_clicks(r, c, set())

        # Count the candidate click itself as 1 step, then add bot steps.
        out = _rollout_with_logic_bot(
            g2,
            max_steps=int(rollout_max_steps),
            allow_mine_triggers=allow_mine_triggers,
            seed=int(seed) + 10_000 * int(i + 1),
        )
        out["steps"] = float(out["steps"] + 1.0)
        outs.append(out)

    # Average across rollouts to reduce label noise.
    def _avg(key: str) -> float:
        return float(sum(float(o.get(key, 0.0) or 0.0) for o in outs) / max(1, len(outs)))

    return {
        "steps": _avg("steps"),
        "cells_opened": _avg("cells_opened"),
        "mines_triggered": _avg("mines_triggered"),
        "game_won": _avg("game_won"),
    }


def _clone_game_fast(game: MinesweeperGame) -> MinesweeperGame:
    """
    Faster alternative to deepcopy for this project.

    I clone only what I need for rollouts:
    - visible board
    - actual board (if initialized)
    - visited grid
    - game state + counters
    """
    g2 = MinesweeperGame(
        height=int(game.height),
        width=int(game.width),
        num_mines=int(game.num_mines),
        seed=None,
        ensure_solvable=bool(getattr(game, "ensure_solvable", False)),
    )

    # Copy hidden board (if present)
    ab = getattr(game, "actual_board", None)
    if ab is not None:
        g2.actual_board = [list(row) for row in ab]

    # Copy visible board + visited
    g2.board = [list(row) for row in getattr(game, "board")]
    g2.visited = [list(row) for row in getattr(game, "visited")]

    # Copy state + bookkeeping
    g2.game_state = getattr(game, "game_state")
    g2.mine_count = int(getattr(game, "mine_count", 0) or 0)
    g2._board_initialized = bool(getattr(game, "_board_initialized", False))
    g2._first_click_position = getattr(game, "_first_click_position", None)

    g2.cells_opened = int(getattr(game, "cells_opened", 0) or 0)
    g2.mines_triggered = int(getattr(game, "mines_triggered", 0) or 0)

    # Preserve any runtime flags used elsewhere
    if hasattr(game, "allow_mine_triggers"):
        setattr(g2, "allow_mine_triggers", bool(getattr(game, "allow_mine_triggers")))

    return g2


def generate_task2_dataset_npz(
    *,
    output_path: str | Path,
    height: int = 22,
    width: int = 22,
    num_mines: int = 80,
    num_games: int = 200,
    states_per_game: int = 32,
    actions_per_state: int = 12,
    rollout_max_steps: int = 512,
    rollouts_per_action: int = 1,
    seed: int = 0,
) -> Path:
    """
    Generate Task 2 data as a single `.npz`.

    Each sample is (visible_board, candidate_action) -> rollout performance.

    Notes:
    - I only target medium difficulty by default (22x22, 80 mines), since Task 2 is specified
      for medium.
    - I keep this generator intentionally simple and reproducible. It's not the fastest, but it
      gets the job done for a course project.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(seed))

    xs: List[np.ndarray] = []
    ars: List[Tuple[int, int]] = []
    y_steps: List[float] = []
    y_cells: List[float] = []
    y_mines: List[float] = []
    y_won: List[int] = []

    # I keep rollouts "traditional": stop when a mine is triggered.
    allow_mine_triggers = False

    for _g in range(int(num_games)):
        game_seed = rng.randint(0, 2**31 - 1)
        game = MinesweeperGame(height=int(height), width=int(width), num_mines=int(num_mines), seed=int(game_seed))
        setattr(game, "allow_mine_triggers", bool(allow_mine_triggers))

        # First click initializes the actual board and is guaranteed safe by this codebase.
        first_click = (rng.randrange(int(height)), rng.randrange(int(width)))
        game.player_clicks(int(first_click[0]), int(first_click[1]), set())
        if game.get_actual_board() is None:
            continue

        # I use LogicBot to walk the game forward and produce realistic intermediate states.
        walker = LogicBot(game, seed=int(game_seed))

        for _s in range(int(states_per_game)):
            if game.get_game_state() != GameState.PROG:
                break

            unrevealed = _unrevealed_cells(game)
            if not unrevealed:
                break

            visible = game.get_visible_board()
            x = visible_to_int8(visible)

            # Candidate actions: random subset + (usually) the logic bot's next click.
            candidates: List[Tuple[int, int]] = []
            k = min(int(actions_per_state), len(unrevealed))
            if k > 0:
                candidates.extend(rng.sample(unrevealed, k=k))

            # Include logic bot's current click choice (if it's valid).
            try:
                a = walker.select_action()
                if a is not None:
                    candidates.append((int(a[0]), int(a[1])))
            except Exception:
                pass

            # De-dup while preserving order.
            seen = set()
            cand_uniq: List[Tuple[int, int]] = []
            for rc in candidates:
                if rc in seen:
                    continue
                seen.add(rc)
                cand_uniq.append((int(rc[0]), int(rc[1])))

            for (r, c) in cand_uniq:
                out = _simulate_action_then_rollout(
                    game,
                    action=(r, c),
                    rollout_max_steps=int(rollout_max_steps),
                    allow_mine_triggers=allow_mine_triggers,
                    seed=rng.randint(0, 2**31 - 1),
                    rollouts_per_action=int(rollouts_per_action),
                )
                xs.append(x)
                ars.append((int(r), int(c)))
                y_steps.append(float(out["steps"]))
                y_cells.append(float(out["cells_opened"]))
                y_mines.append(float(out["mines_triggered"]))
                y_won.append(int(out["game_won"]))

            # Advance the walker by one click step (skipping flag-only steps).
            # Flags don't change the underlying visible board here, so I ignore them.
            for _ in range(32):
                result, action = walker.play_step()
                if result in {"Lost", "Win", "Done"}:
                    break
                if isinstance(action, dict) and action.get("type") == "flag":
                    continue
                break

    if not xs:
        raise RuntimeError("No Task 2 samples generated. Try increasing num_games/states_per_game.")

    x_arr = np.stack(xs).astype(np.int8)
    a_arr = np.asarray(ars, dtype=np.int16)
    ys_arr = np.asarray(y_steps, dtype=np.float32)
    yc_arr = np.asarray(y_cells, dtype=np.float32)
    ym_arr = np.asarray(y_mines, dtype=np.float32)
    yw_arr = np.asarray(y_won, dtype=np.uint8)

    meta = {
        "task": "task2_move_value_prediction",
        "height": int(height),
        "width": int(width),
        "num_mines": int(num_mines),
        "num_games": int(num_games),
        "states_per_game": int(states_per_game),
        "actions_per_state": int(actions_per_state),
        "rollout_max_steps": int(rollout_max_steps),
        "rollouts_per_action": int(rollouts_per_action),
        "rollout_policy": "logic_bot",
        "allow_mine_triggers": bool(allow_mine_triggers),
        "seed": int(seed),
        "num_samples": int(x_arr.shape[0]),
        "target": "y_steps = number of steps survived by logic bot after taking the candidate click (including the click itself)",
    }

    np.savez_compressed(
        out_path,
        x_visible=x_arr,
        action_rc=a_arr,
        y_steps=ys_arr,
        y_cells_opened=yc_arr,
        y_mines_triggered=ym_arr,
        y_game_won=yw_arr,
        meta_json=np.array(json.dumps(meta, ensure_ascii=False)),
    )
    return out_path


def load_task2_npz(path: str | Path) -> Task2NPZ:
    p = Path(path)
    with np.load(p, allow_pickle=False) as data:
        meta_json = str(data["meta_json"].item()) if "meta_json" in data else "{}"
        return Task2NPZ(
            x_visible=data["x_visible"].astype(np.int8),
            action_rc=data["action_rc"].astype(np.int16),
            y_steps=data["y_steps"].astype(np.float32),
            y_cells_opened=data["y_cells_opened"].astype(np.float32),
            y_mines_triggered=data["y_mines_triggered"].astype(np.float32),
            y_game_won=data["y_game_won"].astype(np.uint8),
            meta_json=meta_json,
        )


class Task2Dataset(Dataset):
    """
    Torch dataset wrapper for Task 2 NPZ.
    """

    def __init__(self, npz: Task2NPZ):
        super().__init__()
        self.x = npz.x_visible
        self.a = npz.action_rc
        # I keep the original y_steps as the default target, but I also keep extra rollout
        # signals so I can train on alternative targets if needed.
        self.y_steps = npz.y_steps
        self.y_cells = npz.y_cells_opened
        self.y_mines = npz.y_mines_triggered
        self.y_won = npz.y_game_won

        if self.x.ndim != 3:
            raise ValueError(f"x_visible must be (N,H,W), got {self.x.shape}")
        if self.a.ndim != 2 or self.a.shape[1] != 2:
            raise ValueError(f"action_rc must be (N,2), got {self.a.shape}")
        if self.y_steps.ndim != 1:
            raise ValueError(f"y_steps must be (N,), got {self.y_steps.shape}")
        if self.y_cells.ndim != 1:
            raise ValueError(f"y_cells_opened must be (N,), got {self.y_cells.shape}")
        if self.y_mines.ndim != 1:
            raise ValueError(f"y_mines_triggered must be (N,), got {self.y_mines.shape}")
        if self.y_won.ndim != 1:
            raise ValueError(f"y_game_won must be (N,), got {self.y_won.shape}")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = torch.from_numpy(self.x[idx]).to(torch.int64)  # int64 for one_hot
        a = torch.from_numpy(self.a[idx]).to(torch.int64)
        y_steps = torch.tensor(float(self.y_steps[idx]), dtype=torch.float32)
        y_cells = torch.tensor(float(self.y_cells[idx]), dtype=torch.float32)
        y_mines = torch.tensor(float(self.y_mines[idx]), dtype=torch.float32)
        y_won = torch.tensor(float(self.y_won[idx]), dtype=torch.float32)

        # Backwards-compatible key: "y" == y_steps
        return {
            "x": x,
            "a": a,
            "y": y_steps,
            "y_steps": y_steps,
            "y_cells_opened": y_cells,
            "y_mines_triggered": y_mines,
            "y_game_won": y_won,
        }


