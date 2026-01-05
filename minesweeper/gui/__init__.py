"""
This package is my PyQt GUI for Minesweeper.

I split things by responsibility so `main_window.py` doesn't turn into a 5k-line monster:
- `components`: small reusable widgets (like `StyledButton`)
- `dialogs`: settings/config dialogs
- `constants`: shared presets + helper functions
- `board_manager`: creating/updating the grid of buttons
- `game_manager`: state + replays for bots/loaded files
- `event_handlers`: click/key handlers
"""

# Import non-GUI components that don't require PyQt5
from .constants import PRESETS

# GUI components that require PyQt5 - import on demand
def __getattr__(name):
    if name in ['MinesweeperGUI', 'play_game']:
        from .main_window import MinesweeperGUI, play_game
        return locals()[name]
    elif name == 'StyledButton':
        from .components import StyledButton
        return StyledButton
    elif name == 'GameSettingsDialog':
        from .dialogs import GameSettingsDialog
        return GameSettingsDialog
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'MinesweeperGUI',
    'play_game',
    'StyledButton',
    'GameSettingsDialog',
    'PRESETS',
]

