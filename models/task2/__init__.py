"""
Task 2 (Move Prediction / Actor-Critic).

In Task 2, I treat Minesweeper as a "value of a move" problem:
- input: the current *visible* board + a candidate unrevealed cell to click
- output: a prediction of how long a bot will survive (or how many safe cells it will open)

I use this package for:
- generating supervised regression data by simulating rollouts
- training a critic network
- using the critic to pick better moves than the baseline LogicBot
"""

from .dataset import Task2Dataset, Task2NPZ, generate_task2_dataset_npz, load_task2_npz
from .model import MoveValuePredictor, MoveValuePredictorConfig
from .policy import select_best_unrevealed_by_value

__all__ = [
    "Task2Dataset",
    "Task2NPZ",
    "generate_task2_dataset_npz",
    "load_task2_npz",
    "MoveValuePredictor",
    "MoveValuePredictorConfig",
    "select_best_unrevealed_by_value",
]


