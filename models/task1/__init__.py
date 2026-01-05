"""
Task 1 (Mine Prediction): this is where I keep the model + data utilities.

I kept this package intentionally lightweight (pure Python + NumPy + PyTorch) so I can
train in Colab and still run the same code locally without any weird path issues.
"""

from .encoding import (
    ENC_BLANK,
    ENC_MINE_SHOWN,
    ENC_UNREVEALED,
    visible_to_int8,
)
from .dataset import (
    Task1Dataset,
    generate_task1_dataset_npz,
    load_task1_npz,
)
from .model import MinePredictor
from .policy import select_safest_unrevealed

__all__ = [
    "ENC_UNREVEALED",
    "ENC_BLANK",
    "ENC_MINE_SHOWN",
    "visible_to_int8",
    "Task1Dataset",
    "generate_task1_dataset_npz",
    "load_task1_npz",
    "MinePredictor",
    "select_safest_unrevealed",
]


