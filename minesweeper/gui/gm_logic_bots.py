"""
This file is where I integrate the Logic Bot into the GUI.

I kept this as a separate mixin mostly to keep `game_manager.py` from ballooning. The
end result is the same: I can run the Logic Bot, save the run, and then replay it
action-by-action inside the UI.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import json
import hashlib
from pathlib import Path

from ..game import MinesweeperGame, GameState
from ..logic_bot import LogicBot
from ..board_metrics import count_visible_cells, status_code, total_safe_cells
from .gm_bot_common import make_game_copy, pick_first_click, first_click_state_dict


class LogicBotReplayMixin:
    """
    Mixin: Logic Bot demo execution + loading/scrubbing logic-bot game-state files.

    Expects the parent class (`GameManager`) to provide:
    - core fields: gui, height, width, num_mines, game, original_actual_board, original_first_click
    - shared helpers: _set_game, _update_flagged_array, _normalize_board_cell, _normalize_board_snapshot
    - shared replay helpers: _read_game_states_file, _prepare_loaded_keyframes, _get_loaded_visible_board
    - shared UI helpers: _update_statistics, _set_action_details
    - shared flags helper: _derive_flagged_indices_from_actions
    """

    def run_bot_demo(self):
        """Run bot demonstration."""
        if not getattr(self, "original_actual_board", None):
            return
        # Mark as running so other UI refresh paths don't re-enable buttons mid-run.
        self.bot_running = True

        # Generate bot replay from the beginning of the same actual board,
        # Regardless of current game state (won/lost/in-progress).
        self.bot_states = []
        # I track flags placed by the bot (for export only; UI flags are derived per action index).
        self.bot_flags_on_mines = []

        # Respect the global GUI toggle: when enabled, bots can keep going after mines.
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

        # Perform first click (normalized state dict used by both Logic + NN).
        first_state = first_click_state_dict(game=game_copy, first=first)
        self.bot_states.append(
            {
                "action": first_state.get("action"),
                "board": first_state.get("board"),
                "statistics": game_copy.get_statistics(),
                "game_state": game_copy.get_game_state().value,
                "cells_opened": int(getattr(game_copy, "cells_opened", 0) or 0),
                "mines_triggered": int(getattr(game_copy, "mines_triggered", 0) or 0),
            }
        )

        # LogicBot signature is (game, seed). The actual board is already injected into game_copy.
        bot = LogicBot(game_copy, seed=None)
        # We already applied the first click outside the bot; sync bot's internal bookkeeping.
        try:
            bot.cells_remaining.discard((int(first[0]), int(first[1])))
        except Exception:
            pass
        try:
            bot._update_clue_numbers()
            while bot.make_inferences():
                pass
        except Exception:
            pass

        # Generate bot moves using the LogicBot API (play_step / select_action).
        flags_on_mines_set = set()
        try:
            while True:
                # I stop if we finished a clean win or "done after mines".
                try:
                    if bool(game_copy.check_spaces()):
                        break
                except Exception:
                    pass

                # If continuing after mines is OFF, stop immediately once the run is LOST.
                if (game_copy.get_game_state() != GameState.PROG) and not (allow and game_copy.get_game_state() == GameState.LOST):
                    break

                result, action = bot.play_step()
                if action is None:
                    break

                # If this is a flag action, record it for export.
                if isinstance(action, dict) and str(action.get("type") or "").lower() == "flag":
                    pos = action.get("pos")
                    if isinstance(pos, (list, tuple)) and len(pos) == 2:
                        try:
                            r, c = int(pos[0]), int(pos[1])
                            if (self.original_actual_board
                                and 0 <= r < len(self.original_actual_board)
                                and 0 <= c < len(self.original_actual_board[r])
                                and self.original_actual_board[r][c] == "M"):
                                flags_on_mines_set.add((r, c))
                        except Exception:
                            pass

                # I save state after move
                state = {
                    "action": tuple(action) if isinstance(action, (list, tuple)) else action,
                    "board": [r[:] for r in game_copy.get_visible_board()],
                    "statistics": game_copy.get_statistics(),
                    "game_state": game_copy.get_game_state().value,
                    "cells_opened": game_copy.cells_opened,
                    "mines_triggered": game_copy.mines_triggered,
                }
                self.bot_states.append(state)

                # When continuing after mines, "Lost" is just a signal that a mine was triggered.
                if result in ("Win", "Done"):
                    break
                if result == "Lost" and not bool(getattr(game_copy, "allow_mine_triggers", False)):
                    break
        finally:
            # Always clear the running flag (even on unexpected errors) so the UI doesn't get stuck.
            self.bot_running = False

        # Persist root-level flags_on_mines for export (stable list[list[int]]).
        try:
            self.bot_flags_on_mines = [[int(r), int(c)] for (r, c) in sorted(flags_on_mines_set)]
        except Exception:
            self.bot_flags_on_mines = []

        self.current_state_index = 0
        if self.bot_states:
            self._load_bot_state(0)

    # I keep LogicBot file-integration UI helpers here.
    def _logic_board_hash(self) -> Optional[str]:
        """Hash for the current actual board (same approach as StateCollector)."""
        if not getattr(self, "original_actual_board", None):
            return None
        payload = json.dumps(self.original_actual_board, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _logic_files_for_board(self) -> List[Path]:
        """Find all timestamped logic files for the current board: game_<hash>_logic_*.json"""
        collector = getattr(self.gui, "state_collector", None)
        if collector is None or not hasattr(collector, "output_dir"):
            return []
        h = self._logic_board_hash()
        if not h:
            return []
        out_dir = Path(collector.output_dir)
        return sorted(out_dir.glob(f"game_{h}_logic_*.json"), reverse=True)

    def _latest_logic_file_for_board(self) -> Optional[Path]:
        files = self._logic_files_for_board()
        return files[0] if files else None

    def can_run_logic_bot(self) -> bool:
        """LogicBot is not fully deterministic; allow re-runs whenever a board exists."""
        return bool(getattr(self, "original_actual_board", None))

    def prepare_logic_bot_tab(self):
        """
        Called when entering the Logic Bot tab.
        If a prior run exists, load the newest one for navigation, but keep Run enabled
        (LogicBot can use random fallback moves, so re-running can differ).
        """
        # If a bot is currently running, don't reload/overwrite UI state mid-run.
        if bool(getattr(self, "bot_running", False)):
            if hasattr(self.gui, "run_logic_bot_btn"):
                try:
                    self.gui.run_logic_bot_btn.setEnabled(False)
                except Exception:
                    pass
            self._update_bot_nav_label()
            return

        # If a logic file exists, load the newest one for navigation.
        p = self._latest_logic_file_for_board()
        if p is not None and p.exists():
            self._load_bot_states_from_logic_file(str(p))
        # Always allow running when we have an initialized board.
        if hasattr(self.gui, "run_logic_bot_btn"):
            self.gui.run_logic_bot_btn.setEnabled(bool(getattr(self, "original_actual_board", None)) and not bool(getattr(self, "bot_running", False)))

        # Populate logic bot dropdown (future-proof)
        try:
            from minesweeper.bot_catalog import bots_for_task
        except Exception:
            bots_for_task = None
        if hasattr(self.gui, "logic_bot_selector") and bots_for_task is not None:
            try:
                sel = self.gui.logic_bot_selector
                if sel.count() == 0:
                    try:
                        sel.blockSignals(True)
                    except Exception:
                        pass
                    try:
                        for b in bots_for_task("logic"):
                            sel.addItem(b.name, b.bot_id)
                    finally:
                        try:
                            sel.blockSignals(False)
                        except Exception:
                            pass
            except Exception:
                pass

        # I ensure nav label is sane even if nothing is loaded yet
        self._update_bot_nav_label()

    def _load_bot_states_from_logic_file(self, filepath: str):
        """Load bot states from a saved logic-bot JSON file and enable navigation."""
        try:
            _data, meta, raw_states, _flags = self._read_game_states_file(filepath)
        except Exception:
            return

        # I restore actual board if present
        actual = meta.get("actual_board_initial") if isinstance(meta, dict) else None
        if isinstance(actual, list):
            self.original_actual_board = actual

        # I derive mine count
        mine_ct = None
        try:
            if isinstance(self.original_actual_board, list):
                mine_ct = sum(1 for rr in self.original_actual_board for v in rr if v == "M")
        except Exception:
            mine_ct = None
        if mine_ct is None:
            mine_ct = int(meta.get("num_mines", self.num_mines) or self.num_mines)

        # Dimensions
        h = int(meta.get("height", self.height) or self.height)
        w = int(meta.get("width", self.width) or self.width)

        # I ensure UI dimensions match the loaded bot file.
        if h != self.height or w != self.width:
            self.height, self.width = h, w
            self.gui.height, self.gui.width = h, w
            self._update_flagged_array()
            try:
                self.gui._recreate_board_only(synchronous=True)
            except Exception:
                pass

        # I use the same replay logic as Saved Game States by temporarily wiring metadata/states.
        prev_loaded = getattr(self, "loaded_metadata", None)
        prev_states = getattr(self, "loaded_states", [])
        prev_flags = getattr(self, "loaded_flags_on_mines", [])
        try:
            self.loaded_metadata = meta
            self.loaded_states = [s for s in (raw_states or []) if isinstance(s, dict)]
            self.loaded_flags_on_mines = _flags
            self._prepare_loaded_keyframes()

            def _board_at(i: int):
                return self._get_loaded_visible_board(i)
        finally:
            # I restore Loaded-tab state to avoid cross-tab coupling.
            self.loaded_metadata = prev_loaded
            self.loaded_states = prev_states
            self.loaded_flags_on_mines = prev_flags
            # Keyframe caches are per-loaded-file; clear to be safe.
            self._prepare_loaded_keyframes()

        bot_states: List[Dict] = []
        for i, s in enumerate(raw_states if isinstance(raw_states, list) else []):
            if not isinstance(s, dict):
                continue
            board = s.get("board", []) or _board_at(i)
            action = s.get("action", None)

            counts = count_visible_cells(board)
            total_safe = total_safe_cells(height=int(h), width=int(w), num_mines=int(mine_ct))
            gs = status_code(
                mines_triggered=int(counts.get("mines_shown", 0) or 0),
                safe_opened=int(counts.get("safe_opened", 0) or 0),
                total_safe=int(total_safe),
            )

            bot_states.append({
                "action": tuple(action) if isinstance(action, (list, tuple)) else action,
                "board": self._normalize_board_snapshot(board) if board else [],
                "cells_opened": int(counts.get("safe_opened", 0) or 0),
                "mines_triggered": int(counts.get("mines_shown", 0) or 0),
                "game_state": gs,
                "statistics": {
                    "cells_opened": int(counts.get("safe_opened", 0) or 0),
                    "mines_triggered": int(counts.get("mines_shown", 0) or 0),
                    "game_won": (gs in ("WON", "DONE")),
                },
            })

        self.bot_states = bot_states
        self.current_state_index = 0 if bot_states else -1
        if bot_states:
            self._load_bot_state(0)
        else:
            self._update_bot_nav_label()

    def _update_bot_nav_label(self):
        """Update bot navigation label / jump input under Run Logic Bot."""
        total = len(self.bot_states) if self.bot_states else 0
        current = (self.current_state_index + 1) if self.current_state_index is not None and self.current_state_index >= 0 else 0
        if hasattr(self.gui, "bot_state_nav_label"):
            try:
                self.gui.bot_state_nav_label.setText(f"Action {current} of {total} - Use ← → arrows to navigate")
            except Exception:
                pass
        if hasattr(self.gui, "bot_jump_input"):
            try:
                self.gui.bot_jump_input.setText(str(current))
            except Exception:
                pass

    def jump_to_bot_state(self, action_num: str):
        """Jump to a specific Logic Bot action number (1-based)."""
        try:
            idx = int(action_num) - 1
        except Exception:
            if hasattr(self.gui, "state_details"):
                self.gui.state_details.setText("Invalid input.\nPlease enter a valid action number.")
            return

        if not self.bot_states:
            return
        if idx < 0 or idx >= len(self.bot_states):
            if hasattr(self.gui, "state_details"):
                self.gui.state_details.setText(f"Invalid action.\nAction number must be between 1 and {len(self.bot_states)}.")
            return
        self.current_state_index = idx
        self._load_bot_state(idx)

    def _load_bot_state(self, index: int):
        """Load a specific bot state."""
        if not (0 <= index < len(self.bot_states)):
            return

        state = self.bot_states[index]

        # I update board
        # I ensure single source of truth while replaying bot states
        self._set_game(self.gui.game)
        target_game = self.game

        # I restore flags for this bot action by replaying "flag" actions up to this index.
        try:
            self.flagged_indices = self._derive_flagged_indices_from_actions(self.bot_states, index)
            self.flags_placed = len(self.flagged_indices)
            self._update_flagged_array()
        except Exception:
            self.flagged_indices.clear()
            self.flags_placed = 0
            self._update_flagged_array()
        self.game_started = False

        board = state.get("board", [])
        for row in range(self.height):
            for col in range(self.width):
                if row < len(board) and col < len(board[row]):
                    target_game.board[row][col] = self._normalize_board_cell(board[row][col])
                else:
                    target_game.board[row][col] = "E"

        # I update game state
        target_game.cells_opened = state.get("cells_opened", 0)
        target_game.mines_triggered = state.get("mines_triggered", 0)

        # Per-step state for UI rendering.
        game_state_value = state.get("game_state")
        gs_up = str(game_state_value).strip().upper() if game_state_value is not None else ""
        # DONE means "completed with mines triggered" (no dedicated enum), so treat as WON for UI state.
        if gs_up in ("WON", "DONE"):
            target_game.game_state = GameState.WON
        elif gs_up == "LOST":
            target_game.game_state = GameState.LOST
        else:
            target_game.game_state = GameState.PROG

        self.gui._show_board()
        self.gui._update_all_buttons()
        self._update_statistics()
        self._update_bot_nav_label()

        # Mirror Load Game State details formatting in the right-side details box.
        self._set_action_details(
            source="logic_bot",
            step_index_0=index,
            total_steps=len(self.bot_states),
            action=state.get("action"),
            result=str(game_state_value or target_game.get_game_state().value),
        )

    def navigate_bot_state(self, direction: int):
        """Navigate bot states. direction: 1 for next, -1 for previous."""
        if not self.bot_states:
            return

        new_index = self.current_state_index + direction
        if 0 <= new_index < len(self.bot_states):
            self.current_state_index = new_index
            self._load_bot_state(new_index)


