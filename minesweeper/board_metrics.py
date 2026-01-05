"""
I keep "board metrics" here so the project has one source of truth for:
- how I count progress (safe cells opened)
- what "100% progress" means
- how I map progress + mines_triggered to status codes (PROG/WON/LOST/DONE)

This avoids subtly different implementations across the GUI, state collector, and bot replays.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


VisibleBoard = List[List[Any]]


def _cell(v: Any) -> str:
    """Normalize a visible-board cell to a string, treating None as unrevealed."""
    if v is None:
        return "E"
    try:
        return str(v)
    except Exception:
        return "E"


def total_safe_cells(*, height: int, width: int, num_mines: int) -> int:
    """Total safe cells on the board (used as the denominator for progress)."""
    try:
        h = max(0, int(height))
        w = max(0, int(width))
        m = max(0, int(num_mines))
    except Exception:
        h, w, m = 0, 0, 0
    return max(1, (h * w) - m)


def count_visible_cells(visible_board: VisibleBoard) -> Dict[str, int]:
    """
    Count visible-board symbols.

    Returns a dict:
      - unrevealed: count of "E"
      - mines_shown: count of "M" visible on the board
      - safe_opened: count of everything else (B, numbers, etc.)
    """
    unrevealed = 0
    mines_shown = 0
    safe_opened = 0
    for rr in (visible_board or []):
        for v in (rr or []):
            vv = _cell(v)
            if vv == "E":
                unrevealed += 1
            elif vv == "M":
                mines_shown += 1
            else:
                safe_opened += 1
    return {"unrevealed": int(unrevealed), "mines_shown": int(mines_shown), "safe_opened": int(safe_opened)}


def safe_opened_from_visible(visible_board: VisibleBoard) -> int:
    """Progress numerator: how many *safe* cells are opened, as seen from the visible board."""
    return int(count_visible_cells(visible_board).get("safe_opened", 0) or 0)


def mines_shown_from_visible(visible_board: VisibleBoard) -> int:
    """How many mines are visibly shown as 'M' on the visible board."""
    return int(count_visible_cells(visible_board).get("mines_shown", 0) or 0)


def progress_percent_from_visible(*, visible_board: VisibleBoard, height: int, width: int, num_mines: int) -> int:
    """Compute progress% using the project-wide definition (safe_opened / total_safe)."""
    denom = total_safe_cells(height=height, width=width, num_mines=num_mines)
    opened = safe_opened_from_visible(visible_board)
    try:
        pct = int((float(opened) / float(denom)) * 100.0)
    except Exception:
        pct = 0
    return max(0, min(100, int(pct)))


def progress_complete_from_visible(*, visible_board: VisibleBoard, height: int, width: int, num_mines: int) -> bool:
    """True if progress reached 100% using the project-wide definition."""
    denom = total_safe_cells(height=height, width=width, num_mines=num_mines)
    opened = safe_opened_from_visible(visible_board)
    return bool(int(opened) >= int(denom))


def status_code(*, mines_triggered: int, safe_opened: int, total_safe: int) -> str:
    """
    Single source of truth for file-level / replay-level status codes:
    - WON  if safe_opened >= total_safe and mines_triggered == 0
    - DONE if safe_opened >= total_safe and mines_triggered > 0
    - LOST if mines_triggered > 0 and safe_opened < total_safe
    - PROG otherwise
    """
    try:
        mt = int(mines_triggered) if mines_triggered is not None else 0
    except Exception:
        mt = 0
    try:
        so = int(safe_opened) if safe_opened is not None else 0
    except Exception:
        so = 0
    try:
        ts = max(1, int(total_safe))
    except Exception:
        ts = 1

    if so >= ts and mt == 0:
        return "WON"
    if so >= ts and mt > 0:
        return "DONE"
    if mt > 0:
        return "LOST"
    return "PROG"


def status_code_from_visible(
    *,
    visible_board: VisibleBoard,
    height: int,
    width: int,
    num_mines: int,
    mines_triggered: int,
) -> Tuple[str, Dict[str, int]]:
    """
    Convenience wrapper: compute status code + counts from a visible board.
    Returns (code, counts_dict).
    """
    counts = count_visible_cells(visible_board)
    ts = total_safe_cells(height=height, width=width, num_mines=num_mines)
    code = status_code(mines_triggered=mines_triggered, safe_opened=counts["safe_opened"], total_safe=ts)
    return code, counts


