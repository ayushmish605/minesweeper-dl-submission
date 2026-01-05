"""
This file is where I wire neural-network bots into the GUI replay UX.

My goal here is pretty practical: the Logic Bot tab already has a nice "run + step through +
jump to action" experience, so I reuse that same interaction style for the NN tabs instead
of building a totally separate UI flow.
"""

from __future__ import annotations

from typing import Dict, List

from ..game import MinesweeperGame, GameState
from ..board_metrics import safe_opened_from_visible, total_safe_cells
from .gm_bot_common import make_game_copy, pick_first_click, first_click_state_dict


class NNBotsReplayMixin:
    """
    I use this mixin to keep NN bot replay/navigation code out of `game_manager.py`.

        I wire Task 1/2/3 NN checkpoints into the GUI with the same replay/navigation UX style
        as the Logic Bot tab (run -> step through -> jump to action).
    """

    # Keep shared NN-tab helpers here.
    def _nn_can_run(self) -> bool:
        return bool(getattr(self, "original_actual_board", None))

    def _nn_is_easy_medium_hard(self) -> bool:
        """
        For now, I only support the course presets:
        - 22x22 with mines in {50, 80, 100}
        """
        try:
            return bool(int(self.height) == 22 and int(self.width) == 22 and int(self.num_mines) in (50, 80, 100))
        except Exception:
            return False

    def _nn_can_run_task1(self) -> bool:
        return bool(self._nn_can_run() and self._nn_is_easy_medium_hard())

    def _nn_can_run_task2(self) -> bool:
        # Train Task 2 for Easy/Medium/Hard (same presets as Task 1), so I only enable it there.
        return bool(self._nn_can_run_task1())

    def _nn_can_run_task3(self) -> bool:
        # Train Task 3 for Easy/Medium/Hard (same presets as Task 1), so I only enable it there.
        return bool(self._nn_can_run_task1())

    def _nn_load_state_common(self, state: Dict, index: int, *, source: str, total: int):
        self._set_game(self.gui.game)
        target_game = self.game

        # Don't show flags during NN replays.
        # Leftover flags from LogicBot/Loaded replay can visually cover mines ("M") in the board UI.
        try:
            self.flagged_indices.clear()
        except Exception:
            pass
        try:
            self.flags_placed = 0
        except Exception:
            pass
        try:
            self._update_flagged_array()
        except Exception:
            pass

        board = state.get("board", [])
        for r in range(self.height):
            for c in range(self.width):
                if r < len(board) and c < len(board[r]):
                    target_game.board[r][c] = self._normalize_board_cell(board[r][c])
                else:
                    target_game.board[r][c] = "E"
        target_game.cells_opened = int(state.get("cells_opened", 0) or 0)
        target_game.mines_triggered = int(state.get("mines_triggered", 0) or 0)
        target_game.game_state = GameState.PROG

        self.gui._show_board()
        self.gui._update_all_buttons()
        self._update_statistics()

        self._set_action_details(
            source=source,
            step_index_0=index,
            total_steps=total,
            action=state.get("action"),
            result=str(state.get("game_state", GameState.PROG.value)),
        )

    # Wire Task 1 (mine prediction) into the same replay UX as the LogicBot tab.
    def prepare_nn_mine_tab(self):
        # If I'm already running a bot, I don't fight the loading UX mid-run.
        if bool(getattr(self, "bot_running", False)):
            if hasattr(self.gui, "run_nn_mine_btn"):
                try:
                    self.gui.run_nn_mine_btn.setEnabled(False)
                except Exception:
                    pass
            self._update_nn_mine_nav_label()
            return
        # Fill the dropdown and enable/disable based on current preset + selected model.
        try:
            from minesweeper.bot_catalog import bots_for_task, default_bot_for_preset, get_bot
        except Exception:
            bots_for_task = None
            default_bot_for_preset = None
            get_bot = None

        spec = None
        if hasattr(self.gui, "nn_mine_model_selector") and bots_for_task is not None:
            try:
                sel = self.gui.nn_mine_model_selector
                if sel.count() == 0:
                    try:
                        sel.blockSignals(True)
                    except Exception:
                        pass
                    try:
                        for b in bots_for_task("task1"):
                            sel.addItem(b.name, b.bot_id)
                    finally:
                        try:
                            sel.blockSignals(False)
                        except Exception:
                            pass
                cur_id = sel.currentData()
                if cur_id and get_bot is not None:
                    spec = get_bot("task1", str(cur_id))
                if spec is None and default_bot_for_preset is not None:
                    spec = default_bot_for_preset("task1", height=self.height, width=self.width, num_mines=self.num_mines)
                    if spec is not None:
                        # Select the default model in the UI.
                        for i in range(sel.count()):
                            if sel.itemData(i) == spec.bot_id:
                                try:
                                    sel.blockSignals(True)
                                except Exception:
                                    pass
                                try:
                                    sel.setCurrentIndex(i)
                                finally:
                                    try:
                                        sel.blockSignals(False)
                                    except Exception:
                                        pass
                                break
            except Exception:
                spec = None

        allowed = bool(spec is not None and spec.supports(height=self.height, width=self.width, num_mines=self.num_mines))
        can_run = bool(self._nn_can_run() and allowed and not bool(getattr(self, "bot_running", False)))
        if hasattr(self.gui, "run_nn_mine_btn"):
            self.gui.run_nn_mine_btn.setEnabled(bool(can_run))
        if hasattr(self.gui, "nn_mine_requirements_label") and self.gui.nn_mine_requirements_label is not None:
            try:
                if spec is None:
                    self.gui.nn_mine_requirements_label.setText("Supported presets: Easy / Medium / Hard.")
                else:
                    names = ", ".join(spec.allowed_preset_names())
                    self.gui.nn_mine_requirements_label.setText(f"Supported presets: {names}.")
            except Exception:
                pass
        self._update_nn_mine_nav_label()

    def run_nn_mine_demo(self):
        if not self._nn_can_run_task1():
            if hasattr(self.gui, "state_details"):
                self.gui.state_details.setText(
                    "NN bots are only supported for the Easy / Medium / Hard presets.\n"
                    "Start a New Game and use one of those presets to run this tab."
                )
            return

        # Do lazy imports so the GUI can still open even if torch isn't available yet.
        try:
            import torch
        except Exception:
            if hasattr(self.gui, "state_details"):
                self.gui.state_details.setText("PyTorch not found. Install requirements and restart.")
            return

        try:
            from pathlib import Path
            from models.task1.model import MinePredictor, MinePredictorConfig
            from models.task1.policy import select_safest_unrevealed
        except Exception:
            if hasattr(self.gui, "state_details"):
                self.gui.state_details.setText("Task 1 model code not found. Did you pull the latest `models/task1/` changes?")
            return

        # Choose the checkpoint from the dropdown (bot catalog).
        from minesweeper.bot_catalog import default_bot_for_preset, get_bot

        spec = None
        try:
            sel = getattr(self.gui, "nn_mine_model_selector", None)
            cur_id = sel.currentData() if sel is not None else None
            if cur_id:
                spec = get_bot("task1", str(cur_id))
        except Exception:
            spec = None
        if spec is None:
            spec = default_bot_for_preset("task1", height=self.height, width=self.width, num_mines=self.num_mines)
        if spec is None or not spec.checkpoint_relpath:
            return

        repo_root = Path(__file__).resolve().parents[2]
        ckpt_path = repo_root / str(spec.checkpoint_relpath)
        if not ckpt_path.exists():
            if hasattr(self.gui, "state_details"):
                self.gui.state_details.setText(
                    "Task 1 checkpoint not found.\n\n"
                    f"Expected: {ckpt_path}\n\n"
                    "Train it by running `notebooks/02_train_models_colab.ipynb` (Task 1) and saving checkpoints."
                )
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            mcfg = ckpt.get("model_cfg") or {}
            cfg = MinePredictorConfig(**mcfg)
            model = MinePredictor(cfg).to(device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
        except Exception:
            if hasattr(self.gui, "state_details"):
                self.gui.state_details.setText("Failed to load Task 1 checkpoint.")
            return

        # Generate a replay from the same injected actual board.
        self.nn_mine_states = []

        # Bootstrap the first click and keep a live game_copy for subsequent NN actions.
        allow = bool(getattr(self, "allow_mine_triggers", False))
        game_copy = make_game_copy(
            height=int(self.height),
            width=int(self.width),
            num_mines=int(self.num_mines),
            actual_board=[r[:] for r in self.original_actual_board],
            allow_mine_triggers=bool(allow),
        )
        pref = getattr(self, "original_first_click", None) or getattr(self.game, "_first_click_position", None)
        first = pick_first_click(height=int(self.height), width=int(self.width), actual_board=game_copy.actual_board, preferred=pref)
        if not first:
            return
        self.nn_mine_states.append(first_click_state_dict(game=game_copy, first=first))

        # Compute total safe cells (single source of truth: board_metrics.py).
        try:
            mine_ct = int(getattr(game_copy, "mine_count", 0) or 0)
            if mine_ct <= 0:
                mine_ct = int(self.num_mines)
        except Exception:
            mine_ct = int(self.num_mines)
        total_safe = total_safe_cells(height=int(self.height), width=int(self.width), num_mines=int(mine_ct))

        # Run NN decisions until I clear the board or hit a step cap.
        max_steps = int(self.height) * int(self.width) + 50
        for _ in range(max_steps):
            # Stop when progress reaches 100% (matches GUI progress semantics).
            if safe_opened_from_visible(game_copy.get_visible_board()) >= int(total_safe):
                break

            action = select_safest_unrevealed(model, game_copy.get_visible_board(), device=device, temperature=1.0)
            if action is None:
                break

            r, c = int(action[0]), int(action[1])
            res = game_copy.player_clicks(r, c, set())

            # Derive a UI-friendly state string.
            gs = GameState.PROG.value
            try:
                mines_trig = int(getattr(game_copy, "mines_triggered", 0) or 0)
                if bool(game_copy.check_spaces()):
                    gs = GameState.DONE.value if mines_trig > 0 else GameState.WON.value
                elif mines_trig > 0:
                    gs = GameState.LOST.value
            except Exception:
                gs = GameState.PROG.value

            self.nn_mine_states.append(
                {
                    "action": {"type": "nn", "pos": [r, c], "ckpt": str(ckpt_path.name)},
                    "board": [rr[:] for rr in game_copy.get_visible_board()],
                    "game_state": gs,
                    "cells_opened": int(getattr(game_copy, "cells_opened", 0) or 0),
                    "mines_triggered": int(getattr(game_copy, "mines_triggered", 0) or 0),
                }
            )
            if res == "Lost" and not bool(getattr(game_copy, "allow_mine_triggers", False)):
                # If continuing is OFF, still keep the mine-click action as the final step.
                break

        self.nn_mine_state_index = 0
        self._load_nn_mine_state(0)

    def _update_nn_mine_nav_label(self):
        total = len(getattr(self, "nn_mine_states", []) or [])
        current = (int(getattr(self, "nn_mine_state_index", -1)) + 1) if total else 0
        if hasattr(self.gui, "nn_mine_state_nav_label"):
            self.gui.nn_mine_state_nav_label.setText(f"Action {current} of {total} - Use ← → arrows to navigate")
        if hasattr(self.gui, "nn_mine_jump_input"):
            try:
                self.gui.nn_mine_jump_input.setText(str(current))
            except Exception:
                pass

    def jump_to_nn_mine_state(self, action_num: str):
        try:
            idx = int(action_num) - 1
        except Exception:
            return
        if not self.nn_mine_states:
            return
        if 0 <= idx < len(self.nn_mine_states):
            self.nn_mine_state_index = idx
            self._load_nn_mine_state(idx)

    def navigate_nn_mine_state(self, direction: int):
        if not self.nn_mine_states:
            return
        new_idx = int(getattr(self, "nn_mine_state_index", -1)) + int(direction)
        if 0 <= new_idx < len(self.nn_mine_states):
            self.nn_mine_state_index = new_idx
            self._load_nn_mine_state(new_idx)

    def _load_nn_mine_state(self, index: int):
        if not (0 <= index < len(self.nn_mine_states)):
            return
        self._nn_load_state_common(self.nn_mine_states[index], index, source="nn_mine", total=len(self.nn_mine_states))
        self._update_nn_mine_nav_label()

    # Wire Task 2 (actor/critic move prediction) into the same replay UX as the LogicBot tab.
    def prepare_nn_move_tab(self):
        # If I'm already running a bot, I don't fight the loading UX mid-run.
        if bool(getattr(self, "bot_running", False)):
            if hasattr(self.gui, "run_nn_move_btn"):
                try:
                    self.gui.run_nn_move_btn.setEnabled(False)
                except Exception:
                    pass
            self._update_nn_move_nav_label()
            return
        try:
            from minesweeper.bot_catalog import bots_for_task, default_bot_for_preset, get_bot
        except Exception:
            bots_for_task = None
            default_bot_for_preset = None
            get_bot = None

        spec = None
        if hasattr(self.gui, "nn_move_model_selector") and bots_for_task is not None:
            try:
                sel = self.gui.nn_move_model_selector
                if sel.count() == 0:
                    try:
                        sel.blockSignals(True)
                    except Exception:
                        pass
                    try:
                        for b in bots_for_task("task2"):
                            sel.addItem(b.name, b.bot_id)
                    finally:
                        try:
                            sel.blockSignals(False)
                        except Exception:
                            pass
                cur_id = sel.currentData()
                if cur_id and get_bot is not None:
                    spec = get_bot("task2", str(cur_id))
                if spec is None and default_bot_for_preset is not None:
                    spec = default_bot_for_preset("task2", height=self.height, width=self.width, num_mines=self.num_mines)
                    if spec is not None:
                        for i in range(sel.count()):
                            if sel.itemData(i) == spec.bot_id:
                                try:
                                    sel.blockSignals(True)
                                except Exception:
                                    pass
                                try:
                                    sel.setCurrentIndex(i)
                                finally:
                                    try:
                                        sel.blockSignals(False)
                                    except Exception:
                                        pass
                                break
            except Exception:
                spec = None

        allowed = bool(spec is not None and spec.supports(height=self.height, width=self.width, num_mines=self.num_mines))
        can_run = bool(self._nn_can_run() and allowed and not bool(getattr(self, "bot_running", False)))
        if hasattr(self.gui, "run_nn_move_btn"):
            self.gui.run_nn_move_btn.setEnabled(bool(can_run))
        if hasattr(self.gui, "nn_move_requirements_label") and self.gui.nn_move_requirements_label is not None:
            try:
                if spec is None:
                    self.gui.nn_move_requirements_label.setText("Supported presets: Easy / Medium / Hard.")
                else:
                    names = ", ".join(spec.allowed_preset_names())
                    msg = f"Supported presets: {names}."
                    # If I have per-model results, I show them next to the Run button too.
                    try:
                        extra = str(getattr(spec, "ui_description_text", lambda: "")() or "").strip()
                    except Exception:
                        extra = ""
                    if extra:
                        msg = msg + "\n" + extra
                    self.gui.nn_move_requirements_label.setText(msg)
            except Exception:
                pass
        self._update_nn_move_nav_label()

    def run_nn_move_demo(self):
        # Gate this using the selected model's allowed presets (same style as Task 1).
        if not self._nn_can_run():
            if hasattr(self.gui, "state_details"):
                self.gui.state_details.setText(
                    "NN bots require a started game.\n"
                    "Start a New Game to run this tab."
                )
            return
        # Do lazy imports so the GUI can still open even if torch isn't available yet.
        try:
            import torch
        except Exception:
            if hasattr(self.gui, "state_details"):
                self.gui.state_details.setText("PyTorch not found. Install requirements and restart.")
            return

        try:
            from pathlib import Path
            from minesweeper.logic_bot import LogicBot
            from models.task2.value_map_model import BoardValuePredictor, BoardValuePredictorConfig
            from models.task2.policy import actor_choose_click_value_map
        except Exception:
            if hasattr(self.gui, "state_details"):
                self.gui.state_details.setText("Task 2 model code not found. Did you pull the latest `models/task2/` changes?")
            return

        from minesweeper.bot_catalog import default_bot_for_preset, get_bot
        spec = None
        try:
            sel = getattr(self.gui, "nn_move_model_selector", None)
            cur_id = sel.currentData() if sel is not None else None
            if cur_id:
                spec = get_bot("task2", str(cur_id))
        except Exception:
            spec = None
        if spec is None:
            spec = default_bot_for_preset("task2", height=self.height, width=self.width, num_mines=self.num_mines)
        if spec is None or not spec.supports(height=self.height, width=self.width, num_mines=self.num_mines):
            if hasattr(self.gui, "state_details"):
                self.gui.state_details.setText(
                    "This Task 2 model doesn't match the current board preset.\n"
                    "Pick a Task 2 model that supports this board preset."
                )
            return
        if spec is None or not spec.checkpoint_relpath:
            return

        repo_root = Path(__file__).resolve().parents[2]
        ckpt_path = repo_root / str(spec.checkpoint_relpath)
        if not ckpt_path.exists():
            if hasattr(self.gui, "state_details"):
                self.gui.state_details.setText(
                    "Task 2 checkpoint not found.\n\n"
                    f"Expected: {ckpt_path}\n\n"
                    "Train it by running `notebooks/03_train_task2_colab.ipynb` and saving checkpoints."
                )
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            mcfg = ckpt.get("model_cfg") or {}
            cfg = BoardValuePredictorConfig(**mcfg)
            model = BoardValuePredictor(cfg).to(device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
        except Exception:
            if hasattr(self.gui, "state_details"):
                self.gui.state_details.setText("Failed to load Task 2 checkpoint.")
            return

        # Generate a replay from the same injected actual board.
        self.nn_move_states = []

        allow = bool(getattr(self, "allow_mine_triggers", False))
        game_copy = make_game_copy(
            height=int(self.height),
            width=int(self.width),
            num_mines=int(self.num_mines),
            actual_board=[r[:] for r in self.original_actual_board],
            allow_mine_triggers=bool(allow),
        )
        pref = getattr(self, "original_first_click", None) or getattr(self.game, "_first_click_position", None)
        first = pick_first_click(height=int(self.height), width=int(self.width), actual_board=game_copy.actual_board, preferred=pref)
        if not first:
            return
        self.nn_move_states.append(first_click_state_dict(game=game_copy, first=first))

        # Compute total safe cells (single source of truth: board_metrics.py).
        try:
            mine_ct = int(getattr(game_copy, "mine_count", 0) or 0)
            if mine_ct <= 0:
                mine_ct = int(self.num_mines)
        except Exception:
            mine_ct = int(self.num_mines)
        total_safe = total_safe_cells(height=int(self.height), width=int(self.width), num_mines=int(mine_ct))

        max_steps = int(self.height) * int(self.width) + 50
        for _ in range(max_steps):
            if safe_opened_from_visible(game_copy.get_visible_board()) >= int(total_safe):
                break
            if game_copy.get_game_state() != GameState.PROG:
                if not (bool(getattr(game_copy, "allow_mine_triggers", False)) and game_copy.get_game_state() == GameState.LOST):
                    break

            try:
                from .constants import TASK2_MINE_PENALTY
            except Exception:
                TASK2_MINE_PENALTY = 4.0

            # Use a deterministic seed based on the current step so the run is stable.
            step_seed = int(len(self.nn_move_states) or 0)

            # Match the notebook's behavior:
            # - easy + hard: keep the actor "guess-only" (don't override provably-safe moves)
            # - medium: allow the model to rank safe moves too
            use_model_on_safe_moves = True
            mine_penalty = float(TASK2_MINE_PENALTY)
            try:
                nm = int(self.num_mines)
                if nm <= 50:
                    use_model_on_safe_moves = False
                if nm >= 100:
                    use_model_on_safe_moves = False
                    mine_penalty = 6.0
            except Exception:
                pass

            bot = LogicBot(game_copy, seed=int(step_seed))
            action = actor_choose_click_value_map(
                model=model,
                game=game_copy,
                bot=bot,
                device=device,
                seed=int(step_seed),
                mine_penalty=float(mine_penalty),
                epsilon=0.0,
                top_k=1,
                use_logic_mask=True,
                use_model_on_safe_moves=bool(use_model_on_safe_moves),
            )
            if action is None:
                break
            r, c = int(action[0]), int(action[1])
            res = game_copy.player_clicks(r, c, set())

            gs = GameState.PROG.value
            if res == "Lost":
                gs = GameState.LOST.value
            elif res == "Win":
                gs = GameState.WON.value

            self.nn_move_states.append(
                {
                    "action": {"type": "nn", "pos": [r, c], "ckpt": str(ckpt_path.name)},
                    "board": [rr[:] for rr in game_copy.get_visible_board()],
                    "game_state": gs,
                    "cells_opened": int(getattr(game_copy, "cells_opened", 0) or 0),
                    "mines_triggered": int(getattr(game_copy, "mines_triggered", 0) or 0),
                }
            )

            if res == "Win":
                break
            if res == "Lost" and not bool(getattr(game_copy, "allow_mine_triggers", False)):
                break

        self.nn_move_state_index = 0
        self._load_nn_move_state(0)

    def _update_nn_move_nav_label(self):
        total = len(getattr(self, "nn_move_states", []) or [])
        current = (int(getattr(self, "nn_move_state_index", -1)) + 1) if total else 0
        if hasattr(self.gui, "nn_move_state_nav_label"):
            self.gui.nn_move_state_nav_label.setText(f"Action {current} of {total} - Use ← → arrows to navigate")
        if hasattr(self.gui, "nn_move_jump_input"):
            try:
                self.gui.nn_move_jump_input.setText(str(current))
            except Exception:
                pass

    def jump_to_nn_move_state(self, action_num: str):
        try:
            idx = int(action_num) - 1
        except Exception:
            return
        if not self.nn_move_states:
            return
        if 0 <= idx < len(self.nn_move_states):
            self.nn_move_state_index = idx
            self._load_nn_move_state(idx)

    def navigate_nn_move_state(self, direction: int):
        if not self.nn_move_states:
            return
        new_idx = int(getattr(self, "nn_move_state_index", -1)) + int(direction)
        if 0 <= new_idx < len(self.nn_move_states):
            self.nn_move_state_index = new_idx
            self._load_nn_move_state(new_idx)

    def _load_nn_move_state(self, index: int):
        if not (0 <= index < len(self.nn_move_states)):
            return
        self._nn_load_state_common(self.nn_move_states[index], index, source="nn_move", total=len(self.nn_move_states))
        self._update_nn_move_nav_label()

    # Wire Task 3 (thinking deeper) into the same replay UX as the LogicBot tab.
    def prepare_nn_think_tab(self):
        # If a bot is currently running, don't fight the loading UX mid-run.
        if bool(getattr(self, "bot_running", False)):
            if hasattr(self.gui, "run_nn_think_btn"):
                try:
                    self.gui.run_nn_think_btn.setEnabled(False)
                except Exception:
                    pass
            self._update_nn_think_nav_label()
            return
        try:
            from minesweeper.bot_catalog import bots_for_task, default_bot_for_preset, get_bot
        except Exception:
            bots_for_task = None
            default_bot_for_preset = None
            get_bot = None

        spec = None
        if hasattr(self.gui, "nn_think_model_selector") and bots_for_task is not None:
            try:
                sel = self.gui.nn_think_model_selector
                if sel.count() == 0:
                    try:
                        sel.blockSignals(True)
                    except Exception:
                        pass
                    try:
                        for b in bots_for_task("task3"):
                            sel.addItem(b.name, b.bot_id)
                    finally:
                        try:
                            sel.blockSignals(False)
                        except Exception:
                            pass
                cur_id = sel.currentData()
                if cur_id and get_bot is not None:
                    spec = get_bot("task3", str(cur_id))
                if spec is None and default_bot_for_preset is not None:
                    spec = default_bot_for_preset("task3", height=self.height, width=self.width, num_mines=self.num_mines)
                    if spec is not None:
                        for i in range(sel.count()):
                            if sel.itemData(i) == spec.bot_id:
                                try:
                                    sel.blockSignals(True)
                                except Exception:
                                    pass
                                try:
                                    sel.setCurrentIndex(i)
                                finally:
                                    try:
                                        sel.blockSignals(False)
                                    except Exception:
                                        pass
                                break
            except Exception:
                spec = None

        allowed = bool(spec is not None and spec.supports(height=self.height, width=self.width, num_mines=self.num_mines))
        can_run = bool(self._nn_can_run() and allowed and not bool(getattr(self, "bot_running", False)))
        if hasattr(self.gui, "run_nn_think_btn"):
            self.gui.run_nn_think_btn.setEnabled(bool(can_run))
        if hasattr(self.gui, "nn_think_requirements_label") and self.gui.nn_think_requirements_label is not None:
            try:
                if spec is None:
                    self.gui.nn_think_requirements_label.setText("Supported presets: Easy / Medium / Hard.")
                else:
                    names = ", ".join(spec.allowed_preset_names())
                    self.gui.nn_think_requirements_label.setText(f"Supported presets: {names}.")
            except Exception:
                pass
        self._update_nn_think_nav_label()

    def run_nn_think_demo(self):
        # Gate this using the selected model's allowed presets (same style as Task 1).
        if not self._nn_can_run():
            if hasattr(self.gui, "state_details"):
                self.gui.state_details.setText(
                    "NN bots require a started game.\n"
                    "Start a New Game to run this tab."
                )
            return
        try:
            import torch
        except Exception:
            if hasattr(self.gui, "state_details"):
                self.gui.state_details.setText("PyTorch not found. Install requirements and restart.")
            return

        try:
            from pathlib import Path
            from models.task3.model import ThinkingMinePredictor, ThinkingMinePredictorConfig
            from models.task3.policy import select_safest_unrevealed_thinking
        except Exception:
            if hasattr(self.gui, "state_details"):
                self.gui.state_details.setText("Task 3 model code not found. Did you pull the latest `models/task3/` changes?")
            return

        from minesweeper.bot_catalog import default_bot_for_preset, get_bot
        spec = None
        try:
            sel = getattr(self.gui, "nn_think_model_selector", None)
            cur_id = sel.currentData() if sel is not None else None
            if cur_id:
                spec = get_bot("task3", str(cur_id))
        except Exception:
            spec = None
        if spec is None:
            spec = default_bot_for_preset("task3", height=self.height, width=self.width, num_mines=self.num_mines)
        if spec is None or not spec.supports(height=self.height, width=self.width, num_mines=self.num_mines):
            if hasattr(self.gui, "state_details"):
                self.gui.state_details.setText(
                    "This Task 3 model doesn't match the current board preset.\n"
                    "Pick a Task 3 model that supports this board preset."
                )
            return
        if spec is None or not spec.checkpoint_relpath:
            return

        repo_root = Path(__file__).resolve().parents[2]
        ckpt_path = repo_root / str(spec.checkpoint_relpath)
        if not ckpt_path.exists():
            if hasattr(self.gui, "state_details"):
                self.gui.state_details.setText(
                    "Task 3 checkpoint not found.\n\n"
                    f"Expected: {ckpt_path}\n\n"
                    "Train it by running `notebooks/04_train_task3_colab.ipynb` and saving checkpoints."
                )
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            mcfg = ckpt.get("model_cfg") or {}
            cfg = ThinkingMinePredictorConfig(**mcfg)
            model = ThinkingMinePredictor(cfg).to(device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
        except Exception:
            if hasattr(self.gui, "state_details"):
                self.gui.state_details.setText("Failed to load Task 3 checkpoint.")
            return

        # Generate a replay from the same injected actual board.
        self.nn_think_states = []

        allow = bool(getattr(self, "allow_mine_triggers", False))
        game_copy = make_game_copy(
            height=int(self.height),
            width=int(self.width),
            num_mines=int(self.num_mines),
            actual_board=[r[:] for r in self.original_actual_board],
            allow_mine_triggers=bool(allow),
        )
        pref = getattr(self, "original_first_click", None) or getattr(self.game, "_first_click_position", None)
        first = pick_first_click(height=int(self.height), width=int(self.width), actual_board=game_copy.actual_board, preferred=pref)
        if not first:
            return
        self.nn_think_states.append(first_click_state_dict(game=game_copy, first=first))

        # Compute total safe cells (single source of truth: board_metrics.py).
        try:
            mine_ct = int(getattr(game_copy, "mine_count", 0) or 0)
            if mine_ct <= 0:
                mine_ct = int(self.num_mines)
        except Exception:
            mine_ct = int(self.num_mines)
        total_safe = total_safe_cells(height=int(self.height), width=int(self.width), num_mines=int(mine_ct))

        max_steps = int(self.height) * int(self.width) + 50
        think_steps = int(getattr(cfg, "default_steps", 4) or 4)
        for _ in range(max_steps):
            if safe_opened_from_visible(game_copy.get_visible_board()) >= int(total_safe):
                break

            action = select_safest_unrevealed_thinking(model, game_copy.get_visible_board(), device=device, steps=think_steps, temperature=1.0)
            if action is None:
                break
            r, c = int(action[0]), int(action[1])
            res = game_copy.player_clicks(r, c, set())
            # If I turned off continuing, I stop after the first mine (after recording the step below).

            gs = GameState.PROG.value
            try:
                mines_trig = int(getattr(game_copy, "mines_triggered", 0) or 0)
                if bool(game_copy.check_spaces()):
                    gs = GameState.DONE.value if mines_trig > 0 else GameState.WON.value
                elif mines_trig > 0:
                    gs = GameState.LOST.value
            except Exception:
                gs = GameState.PROG.value

            self.nn_think_states.append(
                {
                    "action": {"type": "nn", "pos": [r, c], "ckpt": str(ckpt_path.name), "think_steps": int(think_steps)},
                    "board": [rr[:] for rr in game_copy.get_visible_board()],
                    "game_state": gs,
                    "cells_opened": int(getattr(game_copy, "cells_opened", 0) or 0),
                    "mines_triggered": int(getattr(game_copy, "mines_triggered", 0) or 0),
                }
            )
            if res == "Lost" and not bool(getattr(game_copy, "allow_mine_triggers", False)):
                break

        self.nn_think_state_index = 0
        self._load_nn_think_state(0)

    def _update_nn_think_nav_label(self):
        total = len(getattr(self, "nn_think_states", []) or [])
        current = (int(getattr(self, "nn_think_state_index", -1)) + 1) if total else 0
        if hasattr(self.gui, "nn_think_state_nav_label"):
            self.gui.nn_think_state_nav_label.setText(f"Action {current} of {total} - Use ← → arrows to navigate")
        if hasattr(self.gui, "nn_think_jump_input"):
            try:
                self.gui.nn_think_jump_input.setText(str(current))
            except Exception:
                pass

    def jump_to_nn_think_state(self, action_num: str):
        try:
            idx = int(action_num) - 1
        except Exception:
            return
        if not self.nn_think_states:
            return
        if 0 <= idx < len(self.nn_think_states):
            self.nn_think_state_index = idx
            self._load_nn_think_state(idx)

    def navigate_nn_think_state(self, direction: int):
        if not self.nn_think_states:
            return
        new_idx = int(getattr(self, "nn_think_state_index", -1)) + int(direction)
        if 0 <= new_idx < len(self.nn_think_states):
            self.nn_think_state_index = new_idx
            self._load_nn_think_state(new_idx)

    def _load_nn_think_state(self, index: int):
        if not (0 <= index < len(self.nn_think_states)):
            return
        self._nn_load_state_common(self.nn_think_states[index], index, source="nn_think", total=len(self.nn_think_states))
        self._update_nn_think_nav_label()


