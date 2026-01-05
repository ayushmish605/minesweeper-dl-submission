"""
Small shared helpers for bot replays (Logic + NN).

I keep these in one place so `gm_logic_bots.py` and `gm_nn_bots.py` don't have to
duplicate the same "bootstrap a replay game" steps:
- create a game_copy with the injected actual board
- pick a consistent first click
- apply the first click and return a normalized first-step dict
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..game import MinesweeperGame, GameState


def mine_count_from_actual(actual_board: List[List[str]], fallback: int) -> int:
    try:
        return int(sum(1 for rr in actual_board for v in rr if v == "M"))
    except Exception:
        return int(fallback)


def make_game_copy(
    *,
    height: int,
    width: int,
    num_mines: int,
    actual_board: List[List[str]],
    allow_mine_triggers: bool,
) -> MinesweeperGame:
    g = MinesweeperGame(int(height), int(width), int(num_mines))
    setattr(g, "allow_mine_triggers", bool(allow_mine_triggers))

    # Inject the mine layout.
    g.actual_board = [r[:] for r in actual_board]
    g._board_initialized = True

    # Keep mine_count consistent for UI stats/progress.
    mine_ct = mine_count_from_actual(actual_board, fallback=int(num_mines))
    try:
        g.mine_count = int(mine_ct)
        g.num_mines = int(mine_ct)
    except Exception:
        pass

    return g


def pick_first_click(
    *,
    height: int,
    width: int,
    actual_board: List[List[str]],
    preferred: Optional[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    if preferred is not None:
        try:
            r, c = int(preferred[0]), int(preferred[1])
            return (r, c)
        except Exception:
            pass

    # Fallback: first safe cell.
    try:
        for r in range(int(height)):
            for c in range(int(width)):
                if actual_board and actual_board[r][c] != "M":
                    return (int(r), int(c))
    except Exception:
        return None
    return None


def first_click_state_dict(*, game: MinesweeperGame, first: Tuple[int, int]) -> Dict[str, Any]:
    """
    Apply the first click to `game` and return a normalized "step 1" state dict
    used by both Logic and NN replays.
    """
    r, c = int(first[0]), int(first[1])
    game.player_clicks(r, c, set())
    return {
        "action": {"type": "deterministic", "pos": [int(r), int(c)]},
        "board": [rr[:] for rr in game.get_visible_board()],
        "game_state": GameState.PROG.value,
        "cells_opened": int(getattr(game, "cells_opened", 0) or 0),
        "mines_triggered": int(getattr(game, "mines_triggered", 0) or 0),
    }


