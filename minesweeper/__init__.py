"""
This is the top-level Minesweeper package I use for the project.

I keep the core game logic importable even when PyQt5 isn't installed, and I
only expose GUI symbols when the GUI dependencies are available.
"""

from .game import MinesweeperGame
from .logic_bot import LogicBot
from .state_collector import StateCollector

# Lazy import for GUI (requires PyQt5)
try:
    from .gui import MinesweeperGUI, play_game
    _GUI_AVAILABLE = True
    _GUI_ERROR = None
except (ImportError, ModuleNotFoundError) as e:
    # GUI not available (PyQt5 missing or other import error)
    _GUI_AVAILABLE = False
    _GUI_ERROR = str(e)
    
    def _gui_not_available(*args, **kwargs):
        raise ImportError(
            f"GUI features require PyQt5. Original error: {_GUI_ERROR}. "
            "Install PyQt5 with 'pip install PyQt5' or use the base MinesweeperGame class instead."
        )
    
    MinesweeperGUI = _gui_not_available
    play_game = _gui_not_available

if _GUI_AVAILABLE:
    __all__ = ['MinesweeperGame', 'LogicBot', 'MinesweeperGUI', 'play_game', 'StateCollector']
else:
    __all__ = ['MinesweeperGame', 'LogicBot', 'StateCollector']

