"""
I keep all bot-tab metadata in one place so the GUI doesn't hardcode tab wiring everywhere.

This file is basically my "registry": it maps each bot tab to the module that implements
its replay/navigation logic, and it also stores the short descriptions shown inside the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class BotTabSpec:
    key: str
    tab_label: str
    title: str
    # A short, scrollable description shown in the tab body.
    description: str
    # Python module that implements the bot replay/navigation mixin.
    impl_module: str


BOT_TABS: List[BotTabSpec] = [
    BotTabSpec(
        key="logic",
        tab_label="Logic Bot",
        title="Logic Bot Demo",
        description=(
            "This is my baseline logic-based Minesweeper agent.\n"
            "It applies deterministic inference rules from clue numbers and falls back to random moves.\n"
            "In this tab, I replay a saved run action-by-action."
        ),
        impl_module="minesweeper.gui.gm_logic_bots",
    ),
    BotTabSpec(
        key="nn_mine",
        tab_label="NN: Mine Prediction",
        title="Neural Net Bot — Mine Prediction",
        description=(
            "Task 1: I predict mine probabilities from the current visible state and click the safest unrevealed cell.\n"
            "I keep the same replay/navigation UX as the Logic Bot tab, but the decisions come from a trained model."
        ),
        impl_module="minesweeper.gui.gm_nn_bots",
    ),
    BotTabSpec(
        key="nn_move",
        tab_label="NN: Move Prediction",
        title="Neural Net Bot — Move Prediction (Actor/Critic)",
        description=(
            "Task 2: I predict survivability of candidate moves (actor/critic style) and use that to pick actions.\n"
            "I mirror the Logic Bot replay/navigation UX here so comparing runs is easy."
        ),
        impl_module="minesweeper.gui.gm_nn_bots",
    ),
    BotTabSpec(
        key="nn_think",
        tab_label="NN: Thinking Deeper",
        title="Neural Net Bot — Thinking Deeper",
        description=(
            "Task 3: I experiment with a model that can 'think longer' via sequential computation.\n"
            "Again, I mirror the Logic Bot replay/navigation UX so I can inspect decisions step-by-step."
        ),
        impl_module="minesweeper.gui.gm_nn_bots",
    ),
]

# Display names for Saved Game States "Mode" column.
# I keep these here so mode labeling is consistent across the GUI.
MODE_DISPLAY_NAMES = {
    "manual": "Manual",
    "logic": "Logic Bot",
    "nn_task1": "NN Task 1",
    "nn_task2": "NN Task 2",
    "nn_task3": "NN Task 3",
}


def mode_display_name(mode_id: str) -> str:
    try:
        key = str(mode_id or "").strip()
    except Exception:
        key = ""
    if not key:
        return "Unknown"
    return str(MODE_DISPLAY_NAMES.get(key, key))


