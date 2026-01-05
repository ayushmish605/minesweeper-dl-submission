"""
These are the GUI constants and small helpers I share across the PyQt code.

I keep difficulty presets here, and I also compute a keyframing stride so saved
game-state JSONs don't explode in size.
"""

import math

# Define shared preset definitions here.
PRESETS = {
    "Easy": {"height": 22, "width": 22, "num_mines": 50},
    "Medium": {"height": 22, "width": 22, "num_mines": 80},
    "Hard": {"height": 22, "width": 22, "num_mines": 100},
}

# Configure game-state saving (Option A: action log + periodic keyframes).

# Scale the *number of keyframes* with board size (cells). The relationship is:
# - 400 cells  -> 10 keyframes
# - 1600 cells -> 20 keyframes
# - 6400 cells -> 30 keyframes
# i.e., every 4× increase in cells adds keyframes.

# These constants define that relation; I derive the actual `keyframe_stride` per game.
KEYFRAMES_BASE_CELLS = 400
KEYFRAMES_BASE_COUNT = 20
KEYFRAMES_LOG_BASE = 4
KEYFRAMES_MIN = 10
KEYFRAMES_MAX = 200


def keyframes_for_cells(num_cells: int) -> int:
    """
    Given the number of cells, I compute how many keyframes I want to store.
    """
    try:
        n = int(num_cells)
    except Exception:
        n = 0
    if n <= 0:
        return int(KEYFRAMES_BASE_COUNT)

# Allow smaller boards to have fewer keyframes as well (ratio < 1 => negative log steps).
    ratio = max(1e-9, float(n) / float(KEYFRAMES_BASE_CELLS))
# Use a base-4 log so that 4× cells -> +BASE_COUNT keyframes.
    steps = math.log(ratio, KEYFRAMES_LOG_BASE)
    kf = float(KEYFRAMES_BASE_COUNT) * (1.0 + steps)
    return int(max(KEYFRAMES_MIN, min(KEYFRAMES_MAX, round(kf))))


def keyframe_stride_for_cells(num_cells: int) -> int:
    """
    Convert desired keyframe count into an action-stride.

    I approximate total actions ~ num_cells. This keeps keyframe density roughly consistent
    across board sizes and works well for bot runs (they tend to open most safe cells).
    """
    try:
        n = int(num_cells)
    except Exception:
        n = 0
    if n <= 0:
        return 25
    kf = max(1, keyframes_for_cells(n))
    return max(1, int(round(float(n) / float(kf))))


# Keep NN-bot GUI config knobs here.

# Task 2 in the GUI can be slow because the critic scores many candidate moves each step.
# Cap how many candidate unrevealed cells are scored per step to keep replay responsive.

# - Larger budget => slower but (usually) better moves
# - Smaller budget => faster but more approximate argmax
TASK2_SEARCH_BUDGET = 128

# If True, I use a tiny deterministic logic pre-filter (only based on visible clues) to:
# - prioritize inferred-safe cells
# - avoid inferred-mine cells
# This is a GUI-side remediation that does not change the model.
TASK2_USE_LOGIC_PREFILTER = True

# If True and "Continue after mine?" is enabled, once a mine has been triggered I stop Task 2
# Do this if I can't infer any guaranteed-safe move from visible clues. This prevents end-of-run behavior
# Where it repeatedly guesses and hits many mines.
TASK2_STOP_AFTER_MINE_IF_UNCERTAIN = True

# Task 2 scoring uses:
# Score = predicted_survival - mine_penalty * P(mine)
# Keep this aligned with the notebook defaults.
TASK2_MINE_PENALTY = 4.0

