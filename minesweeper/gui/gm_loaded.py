"""
This file is my "load + replay" path for saved game-state JSON files.

Because I keyframe board snapshots (to keep files small), I reconstruct intermediate
visible boards during replay so I can scrub through a run smoothly in the GUI.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import json
from pathlib import Path
from datetime import datetime

from ..game import MinesweeperGame, GameState
from ..board_metrics import count_visible_cells, status_code, total_safe_cells


class LoadedReplayMixin:
    """
    Mixin: load/parse game-state JSON files, reconstruct keyframed boards, navigate states,
    delete files/actions, and continue playing from a loaded snapshot.

    Expects the parent class (`GameManager`) to provide:
    - core fields: gui, game, height, width, num_mines
    - shared helpers: _set_game, _update_flagged_array, _derive_flagged_indices_from_actions
    - shared UI helpers: _get_state_label_style, _update_statistics, _set_action_details, save_tab_state
    - other: loaded_* fields are stored on `self`
    """

    # Keep small replay utility primitives here.
    @staticmethod
    def _normalize_board_cell(v):
        """Normalize loaded/bot board entries into the string format the UI expects."""
        if v is None:
            return "E"
        if isinstance(v, bool):
            return "E"
        if isinstance(v, (int, float)):
            return str(int(v))
        return str(v)

    @staticmethod
    def _extract_visible_board(state_dict: Dict) -> List[List[str]]:
        """Extract a visible board snapshot from various schema variants."""
        # Some schema variants wrap most fields under `state`, but keep `board` at the outer level.
        # To stay robust, I check both the inner dict and the outer dict.
        inner = state_dict.get("state", None)
        inner_dict = inner if isinstance(inner, dict) else {}
        outer_dict = state_dict if isinstance(state_dict, dict) else {}

        for d in (inner_dict, outer_dict):
            try:
                b = d.get("board") or d.get("visible_board") or d.get("visible") or d.get("grid")
            except Exception:
                b = None
            if isinstance(b, list) and b:
                return b
        return []

    def _normalize_board_snapshot(self, board: List[List]) -> List[List[str]]:
        """Normalize a 2D board snapshot into the string format the UI expects."""
        out: List[List[str]] = []
        for rr in (board or []):
            row_out: List[str] = []
            for v in (rr or []):
                row_out.append(self._normalize_board_cell(v))
            out.append(row_out)
        return out

    def _prepare_loaded_keyframes(self) -> None:
        """
        Precompute keyframe lookup tables for the currently loaded file.

        Newer files (schema_version>=2) may omit 'board' on most steps. We reconstruct
        those boards on demand by replaying actions from the nearest prior keyframe.
        """
        self._loaded_keyframe_prev = []
        self._loaded_keyframe_boards = {}
        self._loaded_replay_cache = {}
        if not getattr(self, "loaded_states", None):
            return

        prev = -1
        for i, s in enumerate(self.loaded_states):
            board = []
            if isinstance(s, dict):
                board = self._extract_visible_board(s)
            if board:
                prev = i
                self._loaded_keyframe_boards[i] = self._normalize_board_snapshot(board)
            self._loaded_keyframe_prev.append(prev)

    def _action_to_click(self, action):
        """
        Convert various action encodings into either:
        - ("flag", (r,c)) for flags
        - ("click", (r,c)) for clicks
        - (None, None) if not parseable
        """
        if isinstance(action, dict):
            a_type = str(action.get("type") or "").lower()
            pos = action.get("pos")
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                try:
                    r, c = int(pos[0]), int(pos[1])
                    if a_type == "flag":
                        return ("flag", (r, c))
                    # Deterministic/random both map to a click for replay
                    return ("click", (r, c))
                except Exception:
                    return (None, None)
            return (None, None)
        if isinstance(action, (list, tuple)) and len(action) == 2:
            try:
                return ("click", (int(action[0]), int(action[1])))
            except Exception:
                return (None, None)
        return (None, None)

    def _allow_mine_triggers_for_loaded_file(self) -> bool:
        """Best-effort: determine allow_mine_triggers for loaded replay."""
        meta = getattr(self, "loaded_metadata", None) or {}
        if isinstance(meta, dict) and "allow_mine_triggers" in meta:
            try:
                return bool(meta.get("allow_mine_triggers"))
            except Exception:
                return False
        # Historical behavior: manual games stop on a mine; bots default to continuing.
        mode = str((meta or {}).get("mode") or "")
        return mode != "manual"

    def _reconstruct_loaded_visible_board(self, index: int) -> List[List[str]]:
        """Reconstruct a missing visible board by replaying actions from the nearest keyframe."""
        if not self.loaded_states or not (0 <= index < len(self.loaded_states)):
            return []
        meta = getattr(self, "loaded_metadata", None) or {}
        actual = meta.get("actual_board_initial") if isinstance(meta, dict) else None
        if not actual:
            return []
        if not self._loaded_keyframe_prev or len(self._loaded_keyframe_prev) != len(self.loaded_states):
            self._prepare_loaded_keyframes()

        k = self._loaded_keyframe_prev[index] if index < len(self._loaded_keyframe_prev) else -1
        if k < 0:
            return []
        start_board = self._loaded_keyframe_boards.get(k)
        if not start_board:
            return []

        # Cache the exact keyframe board too (cheap win).
        if k not in self._loaded_replay_cache:
            self._loaded_replay_cache[k] = [r[:] for r in start_board]

        # Initialize a replay game from this keyframe snapshot.
        h = int(meta.get("height", len(start_board)) or len(start_board))
        w = int(meta.get("width", len(start_board[0]) if start_board else self.width) or self.width)
        mine_ct = None
        try:
            mine_ct = sum(1 for rr in actual for v in rr if v == "M")
        except Exception:
            mine_ct = None
        if mine_ct is None:
            try:
                mine_ct = int(meta.get("num_mines", self.num_mines) or self.num_mines)
            except Exception:
                mine_ct = int(self.num_mines)

        g = MinesweeperGame(h, w, int(mine_ct))
        setattr(g, "allow_mine_triggers", bool(self._allow_mine_triggers_for_loaded_file()))
        g.actual_board = [r[:] for r in actual]
        g._board_initialized = True
        g.mine_count = int(mine_ct)
        g.num_mines = int(mine_ct)

        # Apply the keyframe visible board snapshot.
        g.board = [r[:] for r in start_board]
        try:
            g.visited = [[0 for _ in range(w)] for _ in range(h)]
            for rr in range(h):
                for cc in range(w):
                    if g.board[rr][cc] != "E":
                        g.visited[rr][cc] = 1
        except Exception:
            pass
        g.cells_opened = sum(1 for rr in g.board for v in rr if v != "E" and v != "M")
        g.mines_triggered = sum(1 for rr in g.board for v in rr if v == "M")

        # Replay actions from k+1 .. index to reach the desired state.
        allow = bool(self._allow_mine_triggers_for_loaded_file())
        for i in range(k + 1, index + 1):
            s = self.loaded_states[i]
            if not isinstance(s, dict):
                continue
            inner = s.get("state", s)
            a = inner.get("action") if isinstance(inner, dict) else None
            if a is None:
                a = s.get("action")
            kind, pos = self._action_to_click(a)
            if kind != "click" or not pos:
                continue
            r, c = pos
            g.player_clicks(r, c, set(), allow_mine_triggers=allow)

        return [r[:] for r in g.board]

    def _get_loaded_visible_board(self, index: int) -> List[List[str]]:
        """Get a visible board snapshot for a loaded state, reconstructing if necessary."""
        if not self.loaded_states or not (0 <= index < len(self.loaded_states)):
            return []
        # Fast path: if this is the final action and the file provides a root-level last_board,
        # Use it directly (avoids reconstruction when keyframing omitted states[-1]["board"]).
        try:
            if index == len(self.loaded_states) - 1:
                lb = getattr(self, "loaded_last_board", None)
                if isinstance(lb, list) and lb:
                    return [r[:] for r in lb]
        except Exception:
            pass
        if index in self._loaded_replay_cache:
            return [r[:] for r in self._loaded_replay_cache[index]]

        s = self.loaded_states[index]
        board = self._extract_visible_board(s) if isinstance(s, dict) else []
        if board:
            out = self._normalize_board_snapshot(board)
            self._loaded_replay_cache[index] = [r[:] for r in out]
            # Small cache cap to avoid unbounded growth
            if len(self._loaded_replay_cache) > 12:
                try:
                    for k in sorted(self._loaded_replay_cache.keys())[:-8]:
                        self._loaded_replay_cache.pop(k, None)
                except Exception:
                    pass
            return out

        out = self._reconstruct_loaded_visible_board(index)
        if out:
            self._loaded_replay_cache[index] = [r[:] for r in out]
            if len(self._loaded_replay_cache) > 12:
                try:
                    for k in sorted(self._loaded_replay_cache.keys())[:-8]:
                        self._loaded_replay_cache.pop(k, None)
                except Exception:
                    pass
        return out

    def _derive_from_visible_board(self, visible_board: List[List[str]]) -> Dict:
        """
        Derive minimal stats + game_state from a visible board snapshot and loaded metadata.
        This replaces the old per-state 'statistics'/'game_state' fields.
        """
        if not visible_board:
            return {"cells_opened": 0, "mines_triggered": 0, "game_state": "PROG"}

        h = len(visible_board)
        w = len(visible_board[0]) if (visible_board and visible_board[0]) else 0

        mine_count = (self.loaded_metadata or {}).get("num_mines", self.num_mines)
        try:
            mine_count = int(mine_count)
        except Exception:
            mine_count = int(self.num_mines)

        counts = count_visible_cells(visible_board)
        ts = total_safe_cells(height=int(h), width=int(w), num_mines=int(mine_count))

        # For loaded snapshots, mines_triggered is represented by mines shown in the board.
        gs = status_code(
            mines_triggered=int(counts.get("mines_shown", 0) or 0),
            safe_opened=int(counts.get("safe_opened", 0) or 0),
            total_safe=int(ts),
        )

        return {
            "cells_opened": int(counts.get("safe_opened", 0) or 0),
            "mines_triggered": int(counts.get("mines_shown", 0) or 0),
            "game_state": str(gs),
        }

    # -----------------------
    # File parsing + load tab
    # -----------------------
    def _read_game_states_file(self, filepath: str):
        """
        Single source of truth for reading a game-states JSON file.
        Returns (data, metadata, states, flags_on_mines).
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        meta = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
        states = data.get("states", []) if isinstance(data.get("states"), list) else []

        flags_raw = data.get("flags_on_mines", [])
        flags: List[List[int]] = []
        if isinstance(flags_raw, list):
            for item in flags_raw:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    try:
                        flags.append([int(item[0]), int(item[1])])
                    except Exception:
                        pass
        return data, meta, states, flags

    def load_game_state_file(self, filepath: str):
        """Load a game state file."""
        try:
            data, meta, raw_states, flags = self._read_game_states_file(filepath)
            self.loaded_metadata = meta
            self.loaded_filepath = filepath
            self.loaded_flags_on_mines = flags
            # Root-level last_board snapshot (if present); used to show the latest action
            # Without reconstructing when keyframing omitted states[-1]["board"].
            try:
                lb = data.get("last_board")
            except Exception:
                lb = None
            if isinstance(lb, list):
                try:
                    self.loaded_last_board = self._normalize_board_snapshot(lb)
                except Exception:
                    self.loaded_last_board = []
            else:
                self.loaded_last_board = []

            # Keep ALL states. Some formats do not include `action` for every step.
            self.loaded_states = [s for s in (raw_states or []) if isinstance(s, dict)]
            # Prepare keyframe lookup for Option A (board snapshots may be sparse).
            self._prepare_loaded_keyframes()

            if not self.loaded_states:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(self.gui, "No States", "This file contains no game states.")
                return

            # Default: load the *actual last action* in the file.
            # With keyframing (schema_version>=2), the last step may omit 'board';
            # We rely on root-level last_board for that.
            last_i = len(self.loaded_states) - 1
            self.loaded_state_index = last_i

            self.viewing_loaded_state = True
            self._load_state_from_file(self.loaded_state_index)
            self._update_nav_label()

            # Stay on Load Game State tab and update the board view there
            self.current_tab_index = 0
            if hasattr(self.gui, "board_tabs"):
                self.gui.board_tabs.setCurrentIndex(0)

        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self.gui, "Error", f"Failed to load game state file: {e}")

    def _get_loaded_file_final_result(self) -> Optional[str]:
        """Return the file-level status code from metadata (WON/LOST/DONE/PROG)."""
        if self._loaded_file_final_result is not None:
            return self._loaded_file_final_result
        if not self.loaded_states:
            self._loaded_file_final_result = None
            return None

        # File-level metadata is the single source of truth.
        meta_gs = (self.loaded_metadata or {}).get("game_state")
        gs = str(meta_gs).strip().upper() if meta_gs is not None else ""
        if gs in ("WON", "LOST", "DONE", "PROG"):
            self._loaded_file_final_result = gs
        else:
            self._loaded_file_final_result = "PROG"
        return self._loaded_file_final_result

    def _update_loaded_state_action_buttons(self, current_game_state: Optional[str] = None):
        """
        Update Load-tab action buttons (Continue Playing / Delete Last Action) based on current selection.
        """
        # Buttons exist only if the UI has created the Load tab.
        continue_btn = getattr(self.gui, "continue_play_button", None)
        delete_btn = getattr(self.gui, "delete_last_action_button", None)
        delete_file_btn = getattr(self.gui, "delete_file_button", None)

        # Default: disabled/hidden when nothing is loaded
        if continue_btn is not None:
            continue_btn.setEnabled(False)
        if delete_btn is not None:
            delete_btn.setEnabled(False)
        if delete_file_btn is not None:
            delete_file_btn.setEnabled(False)

        if not self.loaded_states or self.loaded_state_index < 0:
            return

        # Manual-only restriction (per requirement)
        mode = str((self.loaded_metadata or {}).get("mode") or "")
        is_manual = (mode == "manual")

        # Continue Playing is available if:
        # - manual file
        # - we have actual board metadata to continue safely
        # - we're currently viewing the last action (avoid branching histories in one file)
        # - the selected snapshot is PROG, OR (if "Continue after mine?" is ON) the snapshot is LOST
        file_final = self._get_loaded_file_final_result()
        has_actual = bool((self.loaded_metadata or {}).get("actual_board_initial"))
        cur_state = str(current_game_state).strip().upper() if current_game_state is not None else None
        allow_continue_after_mine = bool(getattr(self, "allow_mine_triggers", False))

        if continue_btn is not None:
            # Always visible when a file is loaded; enabled/disabled based on the rules below.
            continue_btn.setVisible(True)
            # Avoid branching histories in one file: only allow continuing from the last action.
            is_last = (self.loaded_state_index == len(self.loaded_states) - 1)
            can_continue_from_state = bool(cur_state == "PROG" or (allow_continue_after_mine and cur_state == "LOST"))
            continue_btn.setEnabled(bool(is_manual and has_actual and is_last and can_continue_from_state))

        # Delete Last Action is enabled only when currently viewing the last action
        if delete_btn is not None:
            is_last = (self.loaded_state_index == len(self.loaded_states) - 1)
            delete_btn.setEnabled(is_manual and is_last and len(self.loaded_states) > 0 and self.loaded_filepath is not None)

        # Delete File is enabled whenever a file is loaded
        if delete_file_btn is not None:
            delete_file_btn.setEnabled(self.loaded_filepath is not None)

    def delete_loaded_file(self) -> bool:
        """Delete the currently loaded game-states file from disk and clear loaded UI state."""
        if not self.loaded_filepath:
            return False
        try:
            path = Path(self.loaded_filepath)
            try:
                path.unlink()
            except Exception:
                return False

            # Clear loaded UI state
            self.loaded_states.clear()
            self.loaded_state_index = -1
            self.viewing_loaded_state = False
            self.loaded_metadata = None
            self.loaded_filepath = None
            self._loaded_file_final_result = None
            self.loaded_flags_on_mines = []

            # Blank the board display
            self._set_game(self.gui.game)
            for r in range(self.height):
                for c in range(self.width):
                    self.game.board[r][c] = "E"
            self.gui._update_all_buttons()
            self._update_nav_label()
            self._update_loaded_state_action_buttons(None)

            if hasattr(self.gui, "state_label"):
                self.gui.state_label.setText("Game State: Ready")
                self.gui.state_label.setStyleSheet(self._get_state_label_style())
            if hasattr(self.gui, "state_details"):
                self.gui.state_details.setText("Deleted file. Select another saved game to load.")

            return True
        except Exception:
            return False

    def continue_playing_from_loaded_state(self) -> bool:
        """
        Convert the currently viewed loaded state into a playable game and switch to the Playable Game tab.
        Returns True on success.
        """
        if not self.loaded_states or not (0 <= self.loaded_state_index < len(self.loaded_states)):
            return False

        meta = self.loaded_metadata or {}
        if str(meta.get("mode") or "") != "manual":
            return False
        actual_board = meta.get("actual_board_initial")
        if not actual_board:
            # We cannot safely continue without the ground-truth mine layout.
            return False

        state_data = self.loaded_states[self.loaded_state_index]
        inner = state_data.get("state", state_data)
        visible_board = self._get_loaded_visible_board(self.loaded_state_index)
        if not visible_board:
            return False

        allow_continue_after_mine = bool(getattr(self, "allow_mine_triggers", False))
        game_state = str(inner.get("game_state", None) or self._derive_from_visible_board(visible_board).get("game_state", "PROG"))
        gs_up = str(game_state).strip().upper()
        if gs_up != "PROG":
            if not (allow_continue_after_mine and gs_up == "LOST"):
                return False

        derived = self._derive_from_visible_board(visible_board)

        # Prefer metadata dimensions (more reliable) but fall back to visible_board
        h = int(meta.get("height", len(visible_board)) or len(visible_board))
        w = int(meta.get("width", len(visible_board[0]) if visible_board else self.width) or self.width)
        mines = int(meta.get("num_mines", self.num_mines) or self.num_mines)

        # Update dimensions if needed
        if h != self.height or w != self.width or mines != self.num_mines:
            self.height, self.width, self.num_mines = h, w, mines
            self.gui.height, self.gui.width, self.gui.num_mines = h, w, mines
            self._update_flagged_array()
            self.gui._recreate_board_only(synchronous=True)

        # Build a new playable game instance backed by the real mine layout.
        playable = MinesweeperGame(self.height, self.width, self.num_mines)
        try:
            playable.actual_board = [r[:] for r in actual_board]
            playable._board_initialized = True
            # When resuming, we bypass _generate_board(), so we must set mine_count manually
            # (UI and statistics use it for "Mines Remaining").
            try:
                mine_ct = sum(1 for rr in playable.actual_board for v in rr if v == "M")
                playable.mine_count = mine_ct
                playable.num_mines = mine_ct
            except Exception:
                playable.mine_count = int(self.num_mines)
        except Exception:
            # If the underlying game implementation changes, fail safely.
            return False

        # Apply the loaded visible board snapshot.
        for r in range(self.height):
            for c in range(self.width):
                if r < len(visible_board) and c < len(visible_board[r]):
                    playable.board[r][c] = self._normalize_board_cell(visible_board[r][c])
                else:
                    playable.board[r][c] = "E"

        # Derive visited from revealed cells to keep DFS behavior sane when continuing.
        try:
            playable.visited = [[0 for _ in range(self.width)] for _ in range(self.height)]
            for r in range(self.height):
                for c in range(self.width):
                    if playable.board[r][c] != "E":
                        playable.visited[r][c] = 1
        except Exception:
            pass

        playable.cells_opened = int(derived.get("cells_opened", 0) or 0)
        playable.mines_triggered = int(derived.get("mines_triggered", 0) or 0)
        # Purposeful behavior:
        # - The *file* can remain LOST/DONE/WON (it's a record of what happened).
        # - When I press "Continue Playing", I want a playable session that behaves like an
        # In-progress game, as long as continuing is allowed.
        
        # In practice, carrying over GameState.LOST into the live game can make the UI feel
        # "stuck" depending on which guards check `get_game_state()`. So I always restart the
        # Playable session in PROG and rely on mines_triggered + the allow_mine_triggers toggle
        # Do this to reflect that a mine already happened.
        playable.game_state = GameState.PROG
        try:
            setattr(playable, "allow_mine_triggers", bool(allow_continue_after_mine))
        except Exception:
            pass

        # Swap into interactive mode
        self._set_game(playable)
        self.viewing_loaded_state = False
        self.bot_states.clear()
        self.current_state_index = -1
        self.bot_running = False
        self.game_started = True
        self.preview_mode = False

        # Enable Logic Bot for this game (even if it later ends), and store "from the beginning" info.
        try:
            self.original_actual_board = [r[:] for r in playable.actual_board] if playable.actual_board else None
        except Exception:
            self.original_actual_board = None

        # Best-effort: infer original first click from earliest action in loaded_states
        self.original_first_click = None
        for s in self.loaded_states:
            inner_s = s.get("state", s) if isinstance(s, dict) else {}
            a = inner_s.get("action") or (s.get("action") if isinstance(s, dict) else None)
            if isinstance(a, (list, tuple)) and len(a) == 2:
                try:
                    self.original_first_click = (int(a[0]), int(a[1]))
                    break
                except Exception:
                    pass

        if hasattr(self.gui, "board_tabs"):
            self.gui.board_tabs.setTabEnabled(2, bool(self.original_actual_board))
        if hasattr(self.gui, "run_logic_bot_btn"):
            self.gui.run_logic_bot_btn.setEnabled(bool(self.original_actual_board))

        # Reset UI-only flag state (loaded snapshots don't include flags)
        self.flagged_indices.clear()
        self._update_flagged_array()

        # Rehydrate stored correct flags (flags on mines) so resuming restores them.
        self.flags_placed = 0
        for r, c in (self.loaded_flags_on_mines or []):
            try:
                if 0 <= r < self.height and 0 <= c < self.width and playable.board[r][c] == "E":
                    self.flagged[r][c] = True
                    self.flagged_indices.add(r * self.width + c)
                    self.flags_placed += 1
            except Exception:
                continue

        # Moves: each loaded state corresponds to one action
        self.moves = self.loaded_state_index + 1
        self._update_statistics()
        # Let _update_statistics choose the right label (it knows about allow_mine_triggers).
        if hasattr(self.gui, "state_details"):
            self.gui.state_details.setText("Continuing play from a loaded snapshot. Your previous game is still available via saved state files.")

        # Show playable tab WITHOUT triggering restore_tab_state (which would overwrite).
        if hasattr(self.gui, "board_tabs"):
            try:
                self.gui.board_tabs.blockSignals(True)
                self.gui.board_tabs.setCurrentIndex(1)
            finally:
                self.gui.board_tabs.blockSignals(False)
        self.current_tab_index = 1

        self.gui._show_board()
        self.gui._update_all_buttons()

        # Save the resumed game as the playable tab snapshot, so future tab switches don't overwrite it.
        try:
            self.save_tab_state(1)
        except Exception:
            pass

        # If state collection is enabled, continue writing into the SAME loaded game file.
        if getattr(self.gui, "collect_states", False) and getattr(self.gui, "state_collector", None) and self.loaded_filepath:
            try:
                self.gui.state_collector.resume_game_file(self.loaded_filepath)
            except Exception:
                pass

        return True

    def delete_last_loaded_action(self) -> bool:
        """
        Delete the last action/state from the currently loaded game-state file.
        Returns True if something was deleted.
        """
        if not self.loaded_filepath:
            return False
        if str((self.loaded_metadata or {}).get("mode") or "") != "manual":
            return False
        if not self.loaded_states:
            return False
        if self.loaded_state_index != len(self.loaded_states) - 1:
            return False

        try:
            path = Path(self.loaded_filepath)
            with open(path, "r") as f:
                data = json.load(f)

            states = data.get("states", [])
            if not isinstance(states, list) or not states:
                return False

            states.pop()
            now_iso = datetime.now().isoformat()

            if not states:
                # If no states remain, remove the file to avoid a "0 states" artifact.
                try:
                    path.unlink()
                except Exception:
                    # If we can't delete, write back empty (best effort)
                    data["states"] = []
                    data["total_states"] = 0
                    data["last_updated"] = now_iso
                    if isinstance(data.get("metadata"), dict):
                        data["metadata"]["last_updated"] = now_iso
                        data["metadata"]["game_state"] = "PROG"
                    # No states remain, so last_board must be empty too.
                    data["last_board"] = []
                    with open(path, "w") as f:
                        json.dump(data, f, indent=2)

                # Clear loaded UI state
                self.loaded_states.clear()
                self.loaded_state_index = -1
                self.viewing_loaded_state = False
                self.loaded_metadata = None
                self.loaded_filepath = None
                self._loaded_file_final_result = None

                # Blank the board display
                self._set_game(self.gui.game)
                for r in range(self.height):
                    for c in range(self.width):
                        self.game.board[r][c] = "E"
                self.gui._update_all_buttons()
                self._update_nav_label()
                self._update_loaded_state_action_buttons(None)

                if hasattr(self.gui, "state_label"):
                    self.gui.state_label.setText("Loaded State")
                    self.gui.state_label.setStyleSheet(self._get_state_label_style())
                if hasattr(self.gui, "state_details"):
                    self.gui.state_details.setText("Deleted the last remaining action; the file is now removed.")
            else:
                data["states"] = states
                data["total_states"] = len(states)
                data["last_updated"] = now_iso
                if isinstance(data.get("metadata"), dict):
                    data["metadata"]["last_updated"] = now_iso
                    last_state = states[-1] if isinstance(states[-1], dict) else {}
                    last_board = self._extract_visible_board(last_state) if isinstance(last_state, dict) else []

                    # Keyframed files (Option A) may not include 'board' on the final action.
                    # Reconstruct the final visible board so metadata.game_state stays accurate.
                    if not last_board and isinstance(data.get("metadata"), dict):
                        meta = data.get("metadata") or {}
                        actual = meta.get("actual_board_initial")
                        if actual:
                            # Find nearest prior keyframe with a board snapshot.
                            k = -1
                            k_board = None
                            for i in range(len(states) - 1, -1, -1):
                                s2 = states[i] if isinstance(states[i], dict) else {}
                                b2 = self._extract_visible_board(s2) if isinstance(s2, dict) else []
                                if b2:
                                    k = i
                                    k_board = self._normalize_board_snapshot(b2)
                                    break
                            if k >= 0 and k_board:
                                try:
                                    hh = int(meta.get("height", len(k_board)) or len(k_board))
                                    ww = int(meta.get("width", len(k_board[0]) if k_board else self.width) or self.width)
                                except Exception:
                                    hh, ww = len(k_board), (len(k_board[0]) if k_board else self.width)
                                mine_ct = None
                                try:
                                    mine_ct = sum(1 for rr in actual for v in rr if v == "M")
                                except Exception:
                                    mine_ct = None
                                if mine_ct is None:
                                    try:
                                        mine_ct = int(meta.get("num_mines", self.num_mines) or self.num_mines)
                                    except Exception:
                                        mine_ct = int(self.num_mines)

                                g = MinesweeperGame(hh, ww, int(mine_ct))
                                allow = bool(meta.get("allow_mine_triggers")) if "allow_mine_triggers" in meta else (str(meta.get("mode") or "") != "manual")
                                setattr(g, "allow_mine_triggers", bool(allow))
                                g.actual_board = [r[:] for r in actual]
                                g._board_initialized = True
                                g.mine_count = int(mine_ct)
                                g.num_mines = int(mine_ct)
                                g.board = [r[:] for r in k_board]
                                try:
                                    g.visited = [[0 for _ in range(ww)] for _ in range(hh)]
                                    for rr in range(hh):
                                        for cc in range(ww):
                                            if g.board[rr][cc] != "E":
                                                g.visited[rr][cc] = 1
                                except Exception:
                                    pass

                                for i in range(k + 1, len(states)):
                                    s3 = states[i] if isinstance(states[i], dict) else {}
                                    inner3 = s3.get("state", s3) if isinstance(s3, dict) else {}
                                    a3 = inner3.get("action") if isinstance(inner3, dict) else None
                                    if a3 is None and isinstance(s3, dict):
                                        a3 = s3.get("action")
                                    kind, pos = self._action_to_click(a3)
                                    if kind == "click" and pos:
                                        r, c = pos
                                        g.player_clicks(r, c, set(), allow_mine_triggers=bool(allow))

                                last_board = [r[:] for r in g.board]

                    # File-level status is stored as a UI-friendly code.
                    # Recompute file-level status using:
                    # - progress from the reconstructed visible board
                    # - mines_triggered from state fields / replay (NOT by counting visible 'M' cells)
                    # Keep root-level last_board consistent (so the UI can show the latest action
                    # Without reconstructing even when keyframing omits board on the last step).
                    try:
                        data["last_board"] = [r[:] for r in (last_board or [])]
                    except Exception:
                        data["last_board"] = []
                    # Mines_triggered is stored explicitly; prefer last state's value.
                    last_mt = None
                    try:
                        if isinstance(last_state, dict):
                            last_mt = last_state.get("mines_triggered")
                            inner2 = last_state.get("state", last_state) if isinstance(last_state.get("state"), dict) else last_state
                            if isinstance(inner2, dict) and inner2.get("mines_triggered") is not None:
                                last_mt = inner2.get("mines_triggered")
                    except Exception:
                        last_mt = None

                    # If we reconstructed by replaying into `g`, use its counter as the source of truth.
                    try:
                        if "g" in locals() and getattr(g, "mines_triggered", None) is not None:
                            last_mt = int(getattr(g, "mines_triggered", 0) or 0)
                    except Exception:
                        pass

                    try:
                        data["metadata"]["mines_triggered"] = int(last_mt or 0)
                    except Exception:
                        data["metadata"]["mines_triggered"] = int(data["metadata"].get("mines_triggered", 0) or 0)

                    # Determine file-level game_state using progress + mines_triggered.
                    try:
                        meta2 = data.get("metadata") or {}
                        try:
                            h2 = int(meta2.get("height") or 0)
                            w2 = int(meta2.get("width") or 0)
                        except Exception:
                            h2 = 0
                            w2 = 0
                        if h2 <= 0 or w2 <= 0:
                            h2 = len(last_board or [])
                            w2 = len((last_board or [[]])[0]) if (last_board and last_board[0]) else 0
                        total_cells = max(0, int(h2) * int(w2))
                        try:
                            mine_ct2 = int(meta2.get("num_mines", self.num_mines) or self.num_mines)
                        except Exception:
                            mine_ct2 = int(self.num_mines)
                        total_safe = total_safe_cells(height=int(h2), width=int(w2), num_mines=int(mine_ct2))

                        derived2 = self._derive_from_visible_board(last_board or [])
                        safe_opened = int(derived2.get("cells_opened", 0) or 0)
                        mt_val = int(data["metadata"].get("mines_triggered", 0) or 0)
                        prog_100 = bool(int(safe_opened) >= int(total_safe))

                        if prog_100 and mt_val == 0:
                            data["metadata"]["game_state"] = "WON"
                        elif prog_100 and mt_val > 0:
                            data["metadata"]["game_state"] = "DONE"
                        elif mt_val > 0:
                            data["metadata"]["game_state"] = "LOST"
                        else:
                            data["metadata"]["game_state"] = "PROG"
                    except Exception:
                        # If anything goes wrong, fail safely to PROG.
                        try:
                            data["metadata"]["game_state"] = "PROG"
                        except Exception:
                            pass
                with open(path, "w") as f:
                    json.dump(data, f, indent=2)

                # Reload file + UI from disk (resets loaded_state_index to last valid)
                self._loaded_file_final_result = None
                self.load_game_state_file(str(path))

            # Refresh the file list table so the "States" count updates immediately.
            try:
                if hasattr(self.gui, "event_handlers") and hasattr(self.gui, "state_table"):
                    self.gui.event_handlers._populate_state_table()
            except Exception:
                pass

            return True
        except Exception:
            return False

    def _load_state_from_file(self, index: int):
        """Load a specific state from loaded file."""
        if not (0 <= index < len(self.loaded_states)):
            return

        state_data = self.loaded_states[index]

        # Unwrap format variants
        inner = state_data.get("state", state_data)

        visible_board = self._get_loaded_visible_board(index)
        derived = self._derive_from_visible_board(visible_board)
        game_state = inner.get("game_state", None) or derived.get("game_state", "PROG")
        stats = inner.get("statistics", None) or derived
        action = inner.get("action") or state_data.get("action")

        if not visible_board:
            if hasattr(self.gui, "state_label"):
                self.gui.state_label.setText("Loaded State")
                self.gui.state_label.setStyleSheet(self._get_state_label_style())
            if hasattr(self.gui, "state_details"):
                self.gui.state_details.setText(
                    f"Action {index + 1} of {len(self.loaded_states)}\n"
                    "This state has no board snapshot and cannot be reconstructed.\n\n"
                    "Requirements for reconstruction:\n"
                    "- metadata['actual_board_initial'] must exist\n"
                    "- at least one earlier keyframe with 'board' must exist\n"
                    "- actions must be present for intermediate steps"
                )
            return

        # Update dimensions if needed
        new_height = len(visible_board)
        new_width = len(visible_board[0]) if visible_board[0] else self.width

        if new_height != self.height or new_width != self.width:
            self.height, self.width = new_height, new_width
            self.gui.height, self.gui.width = self.height, self.width
            self._update_flagged_array()
            self._set_game(MinesweeperGame(self.height, self.width, self.num_mines))
            self.gui._recreate_board_only(synchronous=True)
        else:
            # Ensure single source of truth: the board renderer reads from gui.game
            self._set_game(self.gui.game)

        # Flags in Saved Game States are derived from action history (same as Logic Bot),
        # PLUS an optional overlay of flags_on_mines (correct flags) depending on show_all_flags.
        self.flagged_indices.clear()
        self._update_flagged_array()

        derived_flags = set()
        try:
            derived_flags = self._derive_flagged_indices_from_actions(self.loaded_states, index)
        except Exception:
            derived_flags = set()

        overlay_flags = set()
        # If show_all_flags is not explicitly set in metadata (older files),
        # Default policy is: manual -> True, all other modes -> False.
        show_all_flags = None
        try:
            show_all_flags = (self.loaded_metadata or {}).get("show_all_flags", None)
        except Exception:
            show_all_flags = None
        if show_all_flags is None:
            try:
                mode = str((self.loaded_metadata or {}).get("mode") or "")
            except Exception:
                mode = ""
            show_all_flags = (mode == "manual")
        else:
            try:
                show_all_flags = bool(show_all_flags)
            except Exception:
                show_all_flags = True

        if show_all_flags:
            for r, c in (self.loaded_flags_on_mines or []):
                try:
                    if 0 <= r < self.height and 0 <= c < self.width:
                        overlay_flags.add(r * self.width + c)
                except Exception:
                    continue

        self.flagged_indices = set(derived_flags) | set(overlay_flags)
        self.flags_placed = len(self.flagged_indices)
        # Sync the 2D flagged grid for any code that still reads it.
        for idx1 in self.flagged_indices:
            try:
                rr = int(idx1) // int(self.width)
                cc = int(idx1) % int(self.width)
                if 0 <= rr < self.height and 0 <= cc < self.width:
                    self.flagged[rr][cc] = True
            except Exception:
                pass

        # Ensure loaded-state mode is consistently treated as non-interactive playback
        self.viewing_loaded_state = True
        self.game_started = False

        target_game = self.gui.game
        for row in range(self.height):
            for col in range(self.width):
                if row < len(visible_board) and col < len(visible_board[row]):
                    target_game.board[row][col] = self._normalize_board_cell(visible_board[row][col])
                else:
                    target_game.board[row][col] = "E"

        self.gui._show_board()
        self.gui._update_all_buttons()

        # Force immediate repaint (Qt can sometimes defer visual updates)
        if hasattr(self.gui, "board_widget") and self.gui.board_widget is not None:
            self.gui.board_widget.update()
            self.gui.board_widget.repaint()

        # Moves are just the action index in the file; cells_opened/mines_triggered are derived.
        self.moves = index + 1
        target_game.cells_opened = int(stats.get("cells_opened", 0) or 0)
        target_game.mines_triggered = int(stats.get("mines_triggered", 0) or 0)
        # Loaded playback does not initialize a real mine layout; ensure mine_count reflects metadata
        # So "Mines Remaining" is meaningful in the stats panel.
        try:
            target_game.mine_count = int((self.loaded_metadata or {}).get("num_mines", target_game.mine_count or self.num_mines))
        except Exception:
            target_game.mine_count = int(self.num_mines)

        gs_up = str(game_state).strip().upper()
        if gs_up == "WON":
            target_game.game_state = GameState.WON
        elif gs_up == "DONE":
            target_game.game_state = GameState.DONE
        elif gs_up == "LOST":
            target_game.game_state = GameState.LOST
        else:
            target_game.game_state = GameState.PROG

        self._update_statistics()

        if hasattr(self.gui, "state_label"):
            self.gui.state_label.setText("Loaded State")
            self.gui.state_label.setStyleSheet(self._get_state_label_style())

        self._set_action_details(
            source="loaded",
            step_index_0=index,
            total_steps=len(self.loaded_states),
            action=action,
            result=str(game_state),
        )

        # Update Load-tab action buttons based on current state and file.
        self._update_loaded_state_action_buttons(str(game_state))

    def _update_nav_label(self):
        """Update navigation label for loaded states."""
        if self.current_tab_index == 0:
            if self.loaded_states:
                total = len(self.loaded_states)
                current = self.loaded_state_index + 1 if self.loaded_state_index >= 0 else 0
                if hasattr(self.gui, "state_nav_label"):
                    self.gui.state_nav_label.setText(f"Action {current} of {total} - Use ← → arrows to navigate")
                if hasattr(self.gui, "state_jump_input"):
                    self.gui.state_jump_input.setText(str(current))
            else:
                if hasattr(self.gui, "state_nav_label"):
                    self.gui.state_nav_label.setText("No state loaded")

    def navigate_loaded_state(self, direction: int):
        """Navigate through loaded states. direction: 1 for next, -1 for previous."""
        if not self.loaded_states:
            return

        new_index = self.loaded_state_index + direction
        if 0 <= new_index < len(self.loaded_states):
            self.loaded_state_index = new_index
            self._load_state_from_file(self.loaded_state_index)
            self._update_nav_label()

    def jump_to_state(self, action_num: str):
        """Jump to specific action number."""
        try:
            index = int(action_num) - 1  # Convert to 0-based
            if 0 <= index < len(self.loaded_states):
                self.loaded_state_index = index
                self._load_state_from_file(index)
                self._update_nav_label()
            else:
                if hasattr(self.gui, "state_details"):
                    self.gui.state_details.setText(
                        f"Invalid action.\nAction number must be between 1 and {len(self.loaded_states)}."
                    )
        except ValueError:
            if hasattr(self.gui, "state_details"):
                self.gui.state_details.setText("Invalid input.\nPlease enter a valid action number.")


