"""
Task 3 (Thinking Deeper).

For Task 3, I keep the same mine-prediction objective as Task 1, but I build a model that
can "think longer" by running a shared reasoning block multiple times.

The key idea:
- one forward pass with steps=1 is a quick guess
- steps=K repeatedly refines the same hidden representation K times
"""

from .model import ThinkingMinePredictor, ThinkingMinePredictorConfig
from .policy import select_safest_unrevealed_thinking

__all__ = ["ThinkingMinePredictor", "ThinkingMinePredictorConfig", "select_safest_unrevealed_thinking"]


