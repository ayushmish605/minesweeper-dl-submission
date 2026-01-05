from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from minesweeper.game import MinesweeperGame, GameState
from minesweeper.logic_bot import LogicBot

from .encoding import (
    mine_mask_from_actual,
    unrevealed_mask_from_visible,
    visible_to_int8,
)


TeacherPolicy = Literal["logic", "random"]


@dataclass(frozen=True)
class Task1NPZ:
    x_visible: np.ndarray  # int8, (N,H,W) values in [-1..9]
    y_mine: np.ndarray  # uint8, (N,H,W) {0,1}
    loss_mask: np.ndarray  # uint8, (N,H,W) {0,1} (1 for unrevealed cells)
    actions: np.ndarray  # int16, (N,2) (row,col) clicked by teacher, -1 if unknown
    meta_json: str


def _logic_teacher_choose_click(bot: LogicBot) -> Optional[Tuple[int, int]]:
    """
    I use LogicBot's internal inference to choose the next CLICK action.
    I intentionally ignore flag actions here, because for Task 1 I'm training a
    "where is the mine?" predictor (and using that to click), not a full flagging agent.
    """
    # Keep bot bookkeeping consistent with any cascade reveals
    bot._sync_sets_with_visible()

    # Update clue numbers + inferences based on current board
    bot._update_clue_numbers()
    while bot.make_inferences():
        pass

    a = bot.select_action()
    if a is None:
        return None
    r, c = a
    return (int(r), int(c))


def _random_teacher_choose_click(game: MinesweeperGame, rng: random.Random) -> Optional[Tuple[int, int]]:
    candidates = [(r, c) for r in range(game.height) for c in range(game.width) if game.board[r][c] == "E"]
    if not candidates:
        return None
    return rng.choice(candidates)


def generate_task1_dataset_npz(
    *,
    output_path: str | Path,
    height: int,
    width: int,
    num_mines: int,
    num_games: int,
    teacher: TeacherPolicy = "logic",
    allow_mine_triggers: bool = True,
    max_clicks_per_game: int = 512,
    seed: int = 0,
) -> Path:
    """
    I generate a Task 1 supervised dataset and save it as a single `.npz`.

    Each sample is one decision point:
    - input: the visible board (what I would actually see as the player)
    - target: mine mask from the ground-truth board (only available during data gen)
    - loss_mask: I only train on unrevealed cells so the loss isn't "free"
    - action: the teacher's chosen click (optional, but useful for debugging/analysis)
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(seed))

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ms: List[np.ndarray] = []
    acts: List[Tuple[int, int]] = []

    for g in range(int(num_games)):
        # I seed each game so this dataset is reproducible.
        game_seed = rng.randint(0, 2**31 - 1)
        game = MinesweeperGame(height=height, width=width, num_mines=num_mines, seed=game_seed)
        setattr(game, "allow_mine_triggers", bool(allow_mine_triggers))

        # I do a first click to initialize `actual_board` (this codebase guarantees it's safe).
        first_click = (rng.randrange(height), rng.randrange(width))
        game.player_clicks(first_click[0], first_click[1], set())
        if game.get_actual_board() is None:
            continue

        bot = LogicBot(game, seed=game_seed) if teacher == "logic" else None

        clicks = 0
        # Important detail: in this codebase, `player_clicks()` can keep going after mine triggers,
        # but the `game_state` flips to LOST and often stays there. For Task 1, I still want states
        # from "after mistakes", so I don't use `game_state` as my hard stop condition.
        while clicks < int(max_clicks_per_game):
            # Stop if all safe cells are opened.
            try:
                if bool(game.check_spaces()):
                    break
            except Exception:
                pass

            # Stop if there are no unrevealed cells left (should coincide with check_spaces()).
            if not any(game.board[r][c] == "E" for r in range(game.height) for c in range(game.width)):
                break

            visible = game.get_visible_board()
            actual = game.get_actual_board()
            if actual is None:
                break

            x = visible_to_int8(visible)
            y = mine_mask_from_actual(actual)
            m = unrevealed_mask_from_visible(visible)

            # Choose teacher action
            if teacher == "logic":
                assert bot is not None
                a = _logic_teacher_choose_click(bot)
            else:
                a = _random_teacher_choose_click(game, rng)
            if a is None:
                break

            xs.append(x)
            ys.append(y)
            ms.append(m)
            acts.append((int(a[0]), int(a[1])))

            # Execute click
            res = game.player_clicks(int(a[0]), int(a[1]), set())
            clicks += 1
            if res == "Lost" and not bool(getattr(game, "allow_mine_triggers", False)):
                break

    if not xs:
        raise RuntimeError("No samples generated. Try increasing num_games or max_clicks_per_game.")

    x_arr = np.stack(xs).astype(np.int8)
    y_arr = np.stack(ys).astype(np.uint8)
    m_arr = np.stack(ms).astype(np.uint8)
    a_arr = np.asarray(acts, dtype=np.int16)

    meta = {
        "task": "task1_mine_prediction",
        "height": int(height),
        "width": int(width),
        "num_mines": int(num_mines),
        "num_games": int(num_games),
        "teacher": str(teacher),
        "allow_mine_triggers": bool(allow_mine_triggers),
        "max_clicks_per_game": int(max_clicks_per_game),
        "seed": int(seed),
        "num_samples": int(x_arr.shape[0]),
        "encoding": {"E": -1, "B_or_0": 0, "1_to_8": "1..8", "M_shown": 9},
        "targets": {"y_mine": "1 if actual cell is a mine", "loss_mask": "1 if visible cell is unrevealed"},
    }

    np.savez_compressed(
        out_path,
        x_visible=x_arr,
        y_mine=y_arr,
        loss_mask=m_arr,
        actions=a_arr,
        meta_json=np.array(json.dumps(meta, ensure_ascii=False)),
    )
    return out_path


def load_task1_npz(path: str | Path) -> Task1NPZ:
    p = Path(path)
    with np.load(p, allow_pickle=False) as data:
        meta_json = str(data["meta_json"].item()) if "meta_json" in data else "{}"
        return Task1NPZ(
            x_visible=data["x_visible"].astype(np.int8),
            y_mine=data["y_mine"].astype(np.uint8),
            loss_mask=data["loss_mask"].astype(np.uint8),
            actions=data["actions"].astype(np.int16) if "actions" in data else np.full((data["x_visible"].shape[0], 2), -1, dtype=np.int16),
            meta_json=meta_json,
        )


class Task1Dataset(Dataset):
    """
    Torch dataset wrapper for Task 1 NPZ.
    """

    def __init__(self, npz: Task1NPZ):
        super().__init__()
        self.x = npz.x_visible
        self.y = npz.y_mine
        self.m = npz.loss_mask

        if self.x.ndim != 3:
            raise ValueError(f"x_visible must be (N,H,W), got {self.x.shape}")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = torch.from_numpy(self.x[idx]).to(torch.int64)  # keep as int64 for one_hot
        y = torch.from_numpy(self.y[idx]).float()
        m = torch.from_numpy(self.m[idx]).float()
        return {"x": x, "y": y, "mask": m}


