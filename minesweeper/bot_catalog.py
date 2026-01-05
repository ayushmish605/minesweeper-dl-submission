"""
Bot catalog (not GUI).

This file is intentionally separate from `minesweeper/gui/`:
- `bot_tab_registry.py` defines GUI tabs
- this file defines which bots exist, what task they belong to, what presets they support,
  and where to load them from (for NN checkpoints)

I keep it as plain Python data so it's easy to extend later (more logic bots, more NN checkpoints).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

try:
    # Reuse the same preset objects the GUI uses (same dict shape).
    from minesweeper.gui.constants import PRESETS
except Exception:
    PRESETS = {
        "Easy": {"height": 22, "width": 22, "num_mines": 50},
        "Medium": {"height": 22, "width": 22, "num_mines": 80},
        "Hard": {"height": 22, "width": 22, "num_mines": 100},
    }


@dataclass(frozen=True)
class BotSpec:
    bot_id: str
    name: str
    task: str  # "task1" | "task2" | "task3" | "logic"
    kind: str  # "logic" | "nn"
    # Restrict presets by listing preset dicts (same shape as PRESETS values).
    # If None, I allow the bot on any board/preset.
    allowed_presets: Optional[Sequence[dict]] = None
    # For NN bots, I store a repo-root-relative path to a .pt file.
    checkpoint_relpath: Optional[str] = None
    # For logic bots, I store an import path like "minesweeper.logic_bot.LogicBot".
    impl_path: Optional[str] = None
    # Optional UI copy: I show this in the GUI near the Run button when this model is selected.
    ui_details: Optional[str] = None

    def supports(self, *, height: int, width: int, num_mines: int) -> bool:
        # Let logic bots work whenever; NN bots are generally preset-restricted.
        if self.allowed_presets is None:
            return True
        try:
            h = int(height)
            w = int(width)
            m = int(num_mines)
        except Exception:
            return False
        for p in self.allowed_presets:
            try:
                if int(p.get("height")) == h and int(p.get("width")) == w and int(p.get("num_mines")) == m:
                    return True
            except Exception:
                continue
        return False

    def allowed_preset_names(self) -> list[str]:
        """
        Human-friendly names for the allowed presets (e.g., Easy/Medium/Hard).
        """
        if self.allowed_presets is None:
            return ["Any"]
        names: list[str] = []
        for ap in self.allowed_presets:
            found = None
            for name, preset in (PRESETS or {}).items():
                if preset == ap:
                    found = str(name)
                    break
            if found is not None:
                names.append(found)
            else:
                try:
                    names.append(f"{ap.get('height')}x{ap.get('width')} mines={ap.get('num_mines')}")
                except Exception:
                    pass
        return names

    def ui_description_text(self) -> str:
        """
        Human-friendly description text for the GUI.
        """
        parts: list[str] = []
        if self.ui_details:
            parts.append(str(self.ui_details).strip())
        return "\n".join([p for p in parts if p])


LOGIC_BOTS: list[BotSpec] = [
    BotSpec(
        bot_id="logic_default",
        name="LogicBot (default)",
        task="logic",
        kind="logic",
        allowed_presets=None,  # Logic bot works whenever
        impl_path="minesweeper.logic_bot.LogicBot",
    )
]


NN_BOTS: list[BotSpec] = [
    # Keep one Task 1 checkpoint per preset.
    BotSpec(
        bot_id="t1_easy",
        name="Task 1 — Mine Predictor (Easy)",
        task="task1",
        kind="nn",
        allowed_presets=(PRESETS.get("Easy") or {"height": 22, "width": 22, "num_mines": 50},),
        checkpoint_relpath="models/task1/checkpoints/task1_easy.pt",
    ),
    BotSpec(
        bot_id="t1_medium",
        name="Task 1 — Mine Predictor (Medium)",
        task="task1",
        kind="nn",
        allowed_presets=(PRESETS.get("Medium") or {"height": 22, "width": 22, "num_mines": 80},),
        checkpoint_relpath="models/task1/checkpoints/task1_medium.pt",
    ),
    BotSpec(
        bot_id="t1_hard",
        name="Task 1 — Mine Predictor (Hard)",
        task="task1",
        kind="nn",
        allowed_presets=(PRESETS.get("Hard") or {"height": 22, "width": 22, "num_mines": 100},),
        checkpoint_relpath="models/task1/checkpoints/task1_hard.pt",
    ),
    # Expose BOTH Task 2 final checkpoints and per-round checkpoints (so I can inspect each round).
    # These round checkpoints are written by `notebooks/03_train_task2_colab.ipynb`.
    BotSpec(
        bot_id="t2_easy_final",
        name="Task 2 — Actor/Critic (Easy, final — r0)",
        task="task2",
        kind="nn",
        allowed_presets=(PRESETS.get("Easy") or {"height": 22, "width": 22, "num_mines": 50},),
        checkpoint_relpath="models/task2/checkpoints/task2_easy.pt",
        ui_details="This loads my final easy checkpoint (round 0).\n(n=80) perfect=0.95 | avg_survival=0.9952 | avg_mines=0.05",
    ),
    BotSpec(
        bot_id="t2_medium_final",
        name="Task 2 — Actor/Critic (Medium, final — r1)",
        task="task2",
        kind="nn",
        allowed_presets=(PRESETS.get("Medium") or {"height": 22, "width": 22, "num_mines": 80},),
        checkpoint_relpath="models/task2/checkpoints/task2_medium.pt",
        ui_details="This loads my final medium checkpoint (round 1).\n(n=80) perfect=0.4125 | avg_survival=0.8050 | avg_mines=1.1375",
    ),
    BotSpec(
        bot_id="t2_hard_final",
        name="Task 2 — Actor/Critic (Hard, final — r0)",
        task="task2",
        kind="nn",
        allowed_presets=(PRESETS.get("Hard") or {"height": 22, "width": 22, "num_mines": 100},),
        checkpoint_relpath="models/task2/checkpoints/task2_hard.pt",
        ui_details="This loads my final hard checkpoint (round 0).\n(n=80) perfect=0.0625 | avg_survival=0.4453 | avg_mines=3.35",
    ),
    # Store Task 2 round checkpoints (v10_target_samples_easy_guess_only).
    BotSpec(
        bot_id="t2_easy_v10_r0",
        name="Task 2 — Round 0 (Easy, logic model)",
        task="task2",
        kind="nn",
        allowed_presets=(PRESETS.get("Easy") or {"height": 22, "width": 22, "num_mines": 50},),
        checkpoint_relpath="models/task2/checkpoints/task2_easy_v10_target_samples_easy_guess_only_round0_logic.pt",
        ui_details="Round 0 (logic model).\n(n=80) perfect=0.95 | avg_survival=0.9952 | avg_mines=0.05",
    ),
    BotSpec(
        bot_id="t2_medium_v10_r0",
        name="Task 2 — Round 0 (Medium, logic model)",
        task="task2",
        kind="nn",
        allowed_presets=(PRESETS.get("Medium") or {"height": 22, "width": 22, "num_mines": 80},),
        checkpoint_relpath="models/task2/checkpoints/task2_medium_v10_target_samples_easy_guess_only_round0_logic.pt",
        ui_details="Round 0 (logic model).\n(n=80) perfect=0.325 | avg_survival=0.7453 | avg_mines=1.25",
    ),
    BotSpec(
        bot_id="t2_medium_v10_r1",
        name="Task 2 — Round 1 (Medium, critic)",
        task="task2",
        kind="nn",
        allowed_presets=(PRESETS.get("Medium") or {"height": 22, "width": 22, "num_mines": 80},),
        checkpoint_relpath="models/task2/checkpoints/task2_medium_v10_target_samples_easy_guess_only_round1_critic.pt",
        ui_details="Round 1 (critic).\n(n=80) perfect=0.4125 | avg_survival=0.8050 | avg_mines=1.1375",
    ),
    BotSpec(
        bot_id="t2_medium_v10_r2",
        name="Task 2 — Round 2 (Medium, critic)",
        task="task2",
        kind="nn",
        allowed_presets=(PRESETS.get("Medium") or {"height": 22, "width": 22, "num_mines": 80},),
        checkpoint_relpath="models/task2/checkpoints/task2_medium_v10_target_samples_easy_guess_only_round2_critic.pt",
        ui_details="Round 2 (critic).\n(n=80) perfect=0.3875 | avg_survival=0.8279 | avg_mines=1.0875",
    ),
    BotSpec(
        bot_id="t2_hard_v10_r0",
        name="Task 2 — Round 0 (Hard, logic model)",
        task="task2",
        kind="nn",
        allowed_presets=(PRESETS.get("Hard") or {"height": 22, "width": 22, "num_mines": 100},),
        checkpoint_relpath="models/task2/checkpoints/task2_hard_v10_target_samples_easy_guess_only_round0_logic.pt",
        ui_details="Round 0 (logic model).\n(n=80) perfect=0.0625 | avg_survival=0.4453 | avg_mines=3.35",
    ),
    BotSpec(
        bot_id="t2_hard_v10_r1",
        name="Task 2 — Round 1 (Hard, critic)",
        task="task2",
        kind="nn",
        allowed_presets=(PRESETS.get("Hard") or {"height": 22, "width": 22, "num_mines": 100},),
        checkpoint_relpath="models/task2/checkpoints/task2_hard_v10_target_samples_easy_guess_only_round1_critic.pt",
        ui_details="Round 1 (critic).\n(n=80) perfect=0.0375 | avg_survival=0.4277 | avg_mines=4.4625",
    ),
    BotSpec(
        bot_id="t2_hard_v10_r2",
        name="Task 2 — Round 2 (Hard, critic)",
        task="task2",
        kind="nn",
        allowed_presets=(PRESETS.get("Hard") or {"height": 22, "width": 22, "num_mines": 100},),
        checkpoint_relpath="models/task2/checkpoints/task2_hard_v10_target_samples_easy_guess_only_round2_critic.pt",
        ui_details="Round 2 (critic).\n(n=80) perfect=0.05 | avg_survival=0.4020 | avg_mines=4.925",
    ),
    # Store Task 3 checkpoints (one per preset, plus tagged versions).
    BotSpec(
        bot_id="t3_easy_v1",
        name="Task 3 — Thinking Mine Predictor (Easy, v1_baseline_15ep_loss_heatmap)",
        task="task3",
        kind="nn",
        allowed_presets=(PRESETS.get("Easy") or {"height": 22, "width": 22, "num_mines": 50},),
        checkpoint_relpath="models/task3/checkpoints/task3_easy_v1_baseline_15ep_loss_heatmap.pt",
        ui_details="This loads my v1 baseline Task 3 checkpoint (15ep).\nI used this run to validate loss-vs-thinking and heatmap evolution first.",
    ),
    BotSpec(
        bot_id="t3_medium_v1",
        name="Task 3 — Thinking Mine Predictor (Medium, v1_baseline_15ep_loss_heatmap)",
        task="task3",
        kind="nn",
        allowed_presets=(PRESETS.get("Medium") or {"height": 22, "width": 22, "num_mines": 80},),
        checkpoint_relpath="models/task3/checkpoints/task3_medium_v1_baseline_15ep_loss_heatmap.pt",
        ui_details="This loads my v1 baseline Task 3 checkpoint (15ep).\nI used this run to validate loss-vs-thinking and heatmap evolution first.",
    ),
    BotSpec(
        bot_id="t3_hard_v1",
        name="Task 3 — Thinking Mine Predictor (Hard, v1_baseline_15ep_loss_heatmap)",
        task="task3",
        kind="nn",
        allowed_presets=(PRESETS.get("Hard") or {"height": 22, "width": 22, "num_mines": 100},),
        checkpoint_relpath="models/task3/checkpoints/task3_hard_v1_baseline_15ep_loss_heatmap.pt",
        ui_details="This loads my v1 baseline Task 3 checkpoint (15ep).\nI used this run to validate loss-vs-thinking and heatmap evolution first.",
    ),
    BotSpec(
        bot_id="t3_easy",
        name="Task 3 — Thinking Mine Predictor (Easy)",
        task="task3",
        kind="nn",
        allowed_presets=(PRESETS.get("Easy") or {"height": 22, "width": 22, "num_mines": 50},),
        checkpoint_relpath="models/task3/checkpoints/task3_easy_v1_baseline_15ep_loss_heatmap.pt",
        ui_details="Convenience alias for my latest available Task 3 easy checkpoint (currently v1 baseline 15ep).",
    ),
    BotSpec(
        bot_id="t3_medium",
        name="Task 3 — Thinking Mine Predictor (Medium)",
        task="task3",
        kind="nn",
        allowed_presets=(PRESETS.get("Medium") or {"height": 22, "width": 22, "num_mines": 80},),
        checkpoint_relpath="models/task3/checkpoints/task3_medium_v1_baseline_15ep_loss_heatmap.pt",
        ui_details="Convenience alias for my latest available Task 3 medium checkpoint (currently v1 baseline 15ep).",
    ),
    BotSpec(
        bot_id="t3_hard",
        name="Task 3 — Thinking Mine Predictor (Hard)",
        task="task3",
        kind="nn",
        allowed_presets=(PRESETS.get("Hard") or {"height": 22, "width": 22, "num_mines": 100},),
        checkpoint_relpath="models/task3/checkpoints/task3_hard_v1_baseline_15ep_loss_heatmap.pt",
        ui_details="Convenience alias for my latest available Task 3 hard checkpoint (currently v1 baseline 15ep).",
    ),
]


def bots_for_task(task: str) -> list[BotSpec]:
    t = str(task).strip().lower()
    if t == "logic":
        return list(LOGIC_BOTS)
    return [b for b in NN_BOTS if str(b.task).lower() == t]


def get_bot(task: str, bot_id: str) -> Optional[BotSpec]:
    for b in bots_for_task(task):
        if b.bot_id == bot_id:
            return b
    return None


def default_bot_for_preset(task: str, *, height: int, width: int, num_mines: int) -> Optional[BotSpec]:
    """
    Pick the first bot that supports the current preset.
    """
    for b in bots_for_task(task):
        if b.supports(height=height, width=width, num_mines=num_mines):
            return b
    # If nothing matches, I fall back to the first bot in the list.
    bs = bots_for_task(task)
    return bs[0] if bs else None


