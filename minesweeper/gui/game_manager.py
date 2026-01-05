# Manage GUI game state and replay/bot plumbing here (game_manager.py).

"""
This is where I keep the GUI-facing game state and all the replay/bot plumbing.

I consolidated older state/bot code paths into one place so `main_window.py` can mostly
focus on UI layout and wiring.
"""

from typing import Dict, List, Optional, Set
from ..game import MinesweeperGame, GameState
from ..board_metrics import progress_percent_from_visible
from .gm_loaded import LoadedReplayMixin
from .gm_logic_bots import LogicBotReplayMixin
from .gm_nn_bots import NNBotsReplayMixin


class GameManager(
    LoadedReplayMixin,
    LogicBotReplayMixin,
    NNBotsReplayMixin,
):
    """I use this to manage game state, bot operations, and tab-specific state for the GUI."""

    def __init__(self, gui):
        self.gui = gui  # Reference to main GUI

        # Track game state.
        self.game: MinesweeperGame = gui.game
        self.height = gui.height
        self.width = gui.width
        self.num_mines = gui.num_mines
        self.cell_size = gui.cell_size

        # Track GUI state (and I initialize it here).
        self.cell_buttons = {}
        self.flagged_indices = set()  # Single source of truth for flagged *widgets*
        self.buttons_clear = set()
        self.moves = 0
        self.flags_placed = 0
        self.flagged = []
        self.game_started = False
        # When I enable this, triggering a mine does not end the run (used for bot/NN evaluation).
        self.allow_mine_triggers = False

        # Initialize the flagged array.
        self._update_flagged_array()

        # Track LogicBot replay state.
        self.bot_states: List[Dict] = []
        self.current_state_index = -1
        self.bot_running = False

        # Track NN bot replay state (Tasks 1–3).
        self.nn_mine_states: List[Dict] = []
        self.nn_mine_state_index = -1
        self.nn_move_states: List[Dict] = []
        self.nn_move_state_index = -1
        self.nn_think_states: List[Dict] = []
        self.nn_think_state_index = -1

        # Track per-tab state.
        self.tab_states = {}  # State for each tab (0=Load Game State, 1=Playable Game, 2=Logic Bot)
        self.current_tab_index = 1  # Default to Playable Game tab

        # Track loaded game-state replays.
        self.loaded_states = []
        self.loaded_state_index = -1
        self.viewing_loaded_state = False
        self.loaded_metadata = None
        self.loaded_filepath: Optional[str] = None
        self._loaded_file_final_result: Optional[str] = None  # Cached per file
        self.loaded_flags_on_mines: List[List[int]] = []
        # Store a root-level last_board snapshot (if present in the loaded JSON file).
        self.loaded_last_board: List[List[str]] = []
        # Store keyframed loaded-state replay caches.
        self._loaded_keyframe_prev: List[int] = []
        self._loaded_keyframe_boards: Dict[int, List[List[str]]] = {}
        self._loaded_replay_cache: Dict[int, List[List[str]]] = {}

        # Store gameplay history for the current session.
        self.game_state_history = []
        self.original_actual_board = None
        self.original_first_click = None
        self.playable_game_state = None
        # When this is True (and on the Playable tab), I render a greyed-out preview until New Game starts.
        self.preview_mode = True

    def _set_game(self, game: MinesweeperGame):
        """
        Single source of truth for the rendered board:
        always keep GameManager.game and GUI.game pointing at the same instance.
        """
        self.game = game
        self.gui.game = game

    # Keep Saved Game States replay + keyframe logic in `gm_loaded.py`.

    def reset_game_state(self, start_game: bool = False):
        """Reset game state with current parameters."""
        # Reset the game instance.
        self._set_game(MinesweeperGame(self.height, self.width, self.num_mines))

        # Reset GUI state.
        self.flagged_indices.clear()
        self.buttons_clear.clear()
        self.moves = 0
        self.flags_placed = 0
        self._update_flagged_array()
        self.game_started = start_game
        self.preview_mode = not start_game
        # Apply the setting to the underlying game instance.
        try:
            setattr(self.game, "allow_mine_triggers", bool(self.allow_mine_triggers))
        except Exception:
            pass
        # Keep the UI checkbox in sync with the current flag.
        if hasattr(self.gui, "continue_after_mine_checkbox") and self.gui.continue_after_mine_checkbox is not None:
            try:
                cb = self.gui.continue_after_mine_checkbox
                cb.blockSignals(True)
                cb.setChecked(bool(self.allow_mine_triggers))
            except Exception:
                pass
            finally:
                try:
                    cb.blockSignals(False)
                except Exception:
                    pass

        # Reset bot and loaded states
        self.bot_states.clear()
        self.current_state_index = -1
        self.bot_running = False
        self.nn_mine_states.clear()
        self.nn_mine_state_index = -1
        self.nn_move_states.clear()
        self.nn_move_state_index = -1
        self.nn_think_states.clear()
        self.nn_think_state_index = -1
        self.viewing_loaded_state = False
        self.loaded_states.clear()
        self.loaded_state_index = -1
        self.loaded_metadata = None
        self.loaded_filepath = None
        self._loaded_file_final_result = None
        self.loaded_flags_on_mines = []
        self.loaded_last_board = []

        # Clear history and tab states
        self.game_state_history.clear()
        self.original_actual_board = None
        self.original_first_click = None
        self.playable_game_state = None
        self.tab_states.clear()

        # Bots tab should only be available once a new game is started (not in preview mode).
        if hasattr(self.gui, 'board_tabs'):
            try:
                self.gui.board_tabs.setTabEnabled(2, bool(start_game))  # Bots
            except Exception:
                pass
        if hasattr(self.gui, 'bot_tab_info'):
            # Keep this label static; dynamic action info is shown in the right-side state_details.
            self.gui.bot_tab_info.setText("Logic Bot replay uses the same Action/Click/Result format in the details pane.")

        # If dimensions changed, we MUST rebuild the board layout (cell_buttons map).
        try:
            expected = int(self.height) * int(self.width)
            if expected > 0 and len(self.cell_buttons) != expected:
                self.gui._recreate_board_only(synchronous=True)
        except Exception:
            pass

        # Reset board state without recreating layout (unless the above forced a rebuild)
        self._reset_board_state(game_started=start_game)

        # Finish game reset
        self._finish_game_reset(start_game)

    def _reset_board_state(self, game_started: bool):
        """Reset board state without recreating the layout."""
        # Note: Game instance should already be reset by caller
        # Reset GUI state only
        self.flagged_indices.clear()
        self.buttons_clear.clear()
        self.moves = 0
        self.flags_placed = 0
        self._update_flagged_array()
        self.game_started = game_started
        self.preview_mode = not game_started
        try:
            setattr(self.game, "allow_mine_triggers", bool(self.allow_mine_triggers))
        except Exception:
            pass

        # Update all buttons to show the new board state
        self.gui._update_all_buttons()

        # Ensure board is visible
        self.gui._show_board()

    def _finish_game_reset(self, start_game: bool):
        """Finish game reset after board recreation."""
        self._update_statistics()
        if hasattr(self.gui, 'state_label'):
            self.gui.state_label.setText("Game State: Ready")
            self.gui.state_label.setStyleSheet(self._get_state_label_style())

        if start_game:
            if hasattr(self.gui, 'board_tabs'):
                self.gui.board_tabs.setCurrentIndex(1)
                self.current_tab_index = 1
            self.gui._show_board()
        else:
            self.gui._show_board_info()

    def apply_settings(self, settings: Dict):
        """Apply new game settings."""
        old_dimensions = (self.height, self.width, self.num_mines)

        # Update dimensions
        self.height = settings['height']
        self.width = settings['width']
        self.num_mines = settings['num_mines']

        # Update GUI references
        self.gui.height = self.height
        self.gui.width = self.width
        self.gui.num_mines = self.num_mines

        # Check if dimensions changed
        new_dimensions = (self.height, self.width, self.num_mines)
        if new_dimensions != old_dimensions:
            # Dimensions changed, need to recreate board layout
            self.reset_game_state(start_game=False)
        else:
            # Only parameters changed, just reset board state
            self._reset_board_state(game_started=False)
            self._finish_game_reset(start_game=False)

        # Ensure preview mode stays on until New Game is pressed.
        self.preview_mode = True
        # When settings change, I treat the board as "not started" (so Bots tab is disabled).
        if hasattr(self.gui, 'board_tabs'):
            try:
                self.gui.board_tabs.setTabEnabled(2, False)
            except Exception:
                pass
        # Keep the toggle checkbox consistent with the manager flag.
        if hasattr(self.gui, "continue_after_mine_checkbox") and self.gui.continue_after_mine_checkbox is not None:
            try:
                cb = self.gui.continue_after_mine_checkbox
                cb.blockSignals(True)
                cb.setChecked(bool(self.allow_mine_triggers))
            except Exception:
                pass
            finally:
                try:
                    cb.blockSignals(False)
                except Exception:
                    pass

        # Update state label for settings change
        if hasattr(self.gui, 'state_label'):
            self.gui.state_label.setText("Settings updated")
            self.gui.state_label.setStyleSheet(self._get_settings_label_style())
        if hasattr(self.gui, "state_details"):
            preset = getattr(self.gui, "preset_name", "Custom")
            self.gui.state_details.setText(
                f"Mode: {preset}\n"
                f"Board: {self.height} x {self.width}   Mines: {self.num_mines}\n"
                "Click 'New Game' to start."
            )

    def _get_state_label_style(self) -> str:
        """Get default state label style."""
        return f"""
            QLabel {{
                color: {self.gui.COLORS['text']};
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                background-color: #2E2E2E;
                border-radius: 5px;
            }}
        """

    def _get_settings_label_style(self) -> str:
        """Get settings change label style."""
        return f"""
            QLabel {{
                color: {self.gui.COLORS['text']};
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                background-color: #FF9800;
                border-radius: 5px;
            }}
        """

    def _update_flagged_array(self):
        """Update flagged array to match current dimensions."""
        self.flagged = [[False for _ in range(self.width)] for _ in range(self.height)]

    def _update_statistics(self):
        """Update statistics display."""
        stats = self.game.get_statistics()
        state = self.game.get_game_state()

        # Avoid fighting the Load Game States tab, which sets its own label ("Loaded State")
        # And uses state_details to show action info.
        if not getattr(self, "viewing_loaded_state", False):
            if hasattr(self.gui, "state_label") and self.gui.state_label:
                if state == GameState.WON:
                    self.gui.state_label.setText("You Win")
                    self.gui.state_label.setStyleSheet(self._get_win_label_style())
                elif state == GameState.LOST:
                    if bool(getattr(self, "allow_mine_triggers", False)):
                        self.gui.state_label.setText("Mine Triggered (continuing)")
                    else:
                        self.gui.state_label.setText("Game Over")
                    self.gui.state_label.setStyleSheet(self._get_loss_label_style())
                else:
                    self.gui.state_label.setText("Game State: In Progress" if self.game_started else "Game State: Ready")
                    self.gui.state_label.setStyleSheet(self._get_state_label_style())

        if hasattr(self.gui, 'moves_label') and self.gui.moves_label:
            self.gui.moves_label.setText(str(self.moves))
        if hasattr(self.gui, 'cells_opened_label') and self.gui.cells_opened_label:
            self.gui.cells_opened_label.setText(str(stats['cells_opened']))
        if hasattr(self.gui, "mines_triggered_label") and self.gui.mines_triggered_label:
            try:
                self.gui.mines_triggered_label.setText(str(int(getattr(self.game, "mines_triggered", 0) or 0)))
            except Exception:
                pass
        # Backwards/forwards compatible: earlier versions used flags_label, current UI uses flags_placed_label.
        flags_widget = getattr(self.gui, "flags_placed_label", None) or getattr(self.gui, "flags_label", None)
        if flags_widget is not None:
            try:
                flags_widget.setText(str(self.flags_placed))
            except Exception:
                pass

        mines_remaining = max(0, self.game.mine_count - self.flags_placed)
        if hasattr(self.gui, 'mines_remaining_label') and self.gui.mines_remaining_label:
            self.gui.mines_remaining_label.setText(str(mines_remaining))

        total_cells = self.height * self.width
        cells_remaining = max(0, total_cells - stats['cells_opened'] - self.flags_placed)
        if hasattr(self.gui, 'cells_remaining_label') and self.gui.cells_remaining_label:
            self.gui.cells_remaining_label.setText(str(cells_remaining))

        # Progress should reflect how much of the SAFE board is revealed.
        # Compute from the currently visible board to avoid drift/over-100% after loss or in loaded/bot playback.
        try:
            mine_ct = int(self.game.mine_count) if getattr(self.game, "mine_count", 0) else int(self.num_mines)
        except Exception:
            mine_ct = int(self.num_mines)
        progress = progress_percent_from_visible(
            visible_board=getattr(self.game, "board", []),
            height=int(self.height),
            width=int(self.width),
            num_mines=int(mine_ct),
        )
        if hasattr(self.gui, 'progress_label') and self.gui.progress_label:
            self.gui.progress_label.setText(f"{progress}%")

        # Small UI note under the stats panel: tab availability rules.
        if hasattr(self.gui, "tab_availability_label") and self.gui.tab_availability_label is not None:
            try:
                bots_ok = bool(self.game_started)
                nn_preset_ok = bool(int(self.height) == 22 and int(self.width) == 22 and int(self.num_mines) in (50, 80, 100))
                self.gui.tab_availability_label.setText(
                    "Tab availability:\n"
                    f"- Bots tab: {'enabled' if bots_ok else 'disabled'} (only after I start a New Game)\n"
                    f"- NN bots: {'enabled' if nn_preset_ok else 'disabled'} (only for Easy/Medium/Hard presets)"
                )
            except Exception:
                pass

    def _set_action_details(
        self,
        *,
        source: str,
        step_index_0: int,
        total_steps: int,
        action,
        result: str,
    ):
        """
        Unified rendering for the right-side details box.
        Format matches Load Game States:
          Action X of Y
          Click: (r, c)
          Result: <...>

        This is intended to be reused for Logic Bot and future NN replays.
        """
        if not hasattr(self.gui, "state_details") or self.gui.state_details is None:
            return
        current = int(step_index_0) + 1
        total = int(total_steps)

        header = f"Action {current} of {total}"

        # Support both legacy click tuples and richer action dicts (e.g., LogicBot flags).
        if isinstance(action, dict):
            a_type = str(action.get("type") or "").lower()
            pos = action.get("pos")
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                try:
                    r, c = int(pos[0]), int(pos[1])
                    if a_type == "flag":
                        label = "Flag"
                    elif a_type in ("random", "deterministic"):
                        label = f"Click ({a_type})"
                    else:
                        label = "Click"
                    self.gui.state_details.setText(
                        f"{header}\n"
                        f"{label}: ({r}, {c})\n"
                        f"Result: {result}"
                    )
                    return
                except Exception:
                    pass
            self.gui.state_details.setText(f"{header}\nResult: {result}")
            return

        if action and isinstance(action, (list, tuple)) and len(action) == 2:
            try:
                r, c = int(action[0]), int(action[1])
                self.gui.state_details.setText(
                    f"{header}\n"
                    f"Click: ({r}, {c})\n"
                    f"Result: {result}"
                )
                return
            except Exception:
                pass

        self.gui.state_details.setText(f"{header}\nResult: {result}")

    def _derive_flagged_indices_from_actions(self, states: List[Dict], up_to_index: int) -> Set[int]:
        """
        Derive UI flags from a sequence of actions up to a given index.

        We keep per-state JSON minimal (action + board) by reconstructing flags for playback.
        """
        # Interpret "flag" actions as toggles so we can support both place/remove semantics.
        out: Set[int] = set()
        if not states:
            return out
        try:
            end = min(int(up_to_index), len(states) - 1)
        except Exception:
            end = len(states) - 1
        for s in states[: end + 1]:
            if not isinstance(s, dict):
                continue
            inner = s.get("state", s) if isinstance(s.get("state"), dict) else s
            a = inner.get("action") if isinstance(inner, dict) else s.get("action")
            if not isinstance(a, dict):
                continue
            if str(a.get("type") or "").lower() != "flag":
                continue
            pos = a.get("pos")
            if not (isinstance(pos, (list, tuple)) and len(pos) == 2):
                continue
            try:
                r, c = int(pos[0]), int(pos[1])
                if 0 <= r < self.height and 0 <= c < self.width:
                    idx = r * self.width + c
                    if idx in out:
                        out.discard(idx)
                    else:
                        out.add(idx)
            except Exception:
                pass
        return out

    def _get_win_label_style(self) -> str:
        return f"""
            QLabel {{
                color: {self.gui.COLORS['text']};
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                background-color: #4CAF50;
                border-radius: 5px;
            }}
        """

    def _get_loss_label_style(self) -> str:
        return f"""
            QLabel {{
                color: {self.gui.COLORS['text']};
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                background-color: #D32F2F;
                border-radius: 5px;
            }}
        """

    def save_tab_state(self, tab_index: int):
        """Save current state for a specific tab."""
        if tab_index < 0:
            return
        # Do not snapshot the Load Game State tab board; it should always be driven by
        # `loaded_states[loaded_state_index]` to avoid stale restores fighting navigation.
        if tab_index == 0:
            return

        visible_board = [row[:] for row in self.game.get_visible_board()] if hasattr(self.game, 'get_visible_board') else None
        actual_board = self.game.get_actual_board()
        if actual_board:
            actual_board_copy = [row[:] for row in actual_board]
        else:
            actual_board_copy = None

        state_dict = {
            'visible_board': visible_board,
            'actual_board': actual_board_copy,
            'game_board': [row[:] for row in self.game.board],
            'flagged': [row[:] for row in self.flagged],
            'flagged_indices': list(self.flagged_indices),
            'buttons_clear': set(self.buttons_clear),
            'moves': self.moves,
            'flags_placed': self.flags_placed,
            'statistics': self.game.get_statistics(),
            'game_state': self.game.get_game_state().value if hasattr(self.game.get_game_state(), 'value') else self.game.get_game_state(),
            'cells_opened': getattr(self.game, 'cells_opened', 0),
            'mines_triggered': getattr(self.game, 'mines_triggered', 0),
            'board_dimensions': (self.height, self.width, self.num_mines),
            'game_started': self.game_started,
        }

        state_dict.update({
            'viewing_loaded_state': self.viewing_loaded_state,
            'loaded_state_index': self.loaded_state_index,
            'current_state_index': self.current_state_index,
        })

        self.tab_states[tab_index] = state_dict

    def restore_tab_state(self, tab_index: int):
        """Restore state for a specific tab."""
        # If the board isn't built yet, do not try to "restore UI".
        # Check the actual button map that BoardManager populates.
        gm = self.gui.game_manager if hasattr(self.gui, "game_manager") else None
        if (gm is None or not getattr(gm, "cell_buttons", None)) and tab_index in (0, 1, 2):
            self.current_tab_index = tab_index
            return

        # Load Game State tab is driven directly by loaded file navigation, not snapshots.
        if tab_index == 0:
            self.current_tab_index = 0
            if self.loaded_states and self.loaded_state_index >= 0:
                self._load_state_from_file(self.loaded_state_index)
                self._update_nav_label()
                self.gui._show_board()
            else:
                self.gui._show_board_info("Select a game state file from the table to load.")
            return

        if tab_index not in self.tab_states:
            # Initialize based on tab type
            if tab_index == 0:  # Load Game State
                if self.loaded_states and self.loaded_state_index >= 0:
                    self.gui._show_board()
                else:
                    self.gui._show_board_info("Select a game state file from the table to load.")
            elif tab_index == 1:  # Playable Game
                self.gui._show_board()
            elif tab_index == 2:  # Logic Bot
                if self.bot_states and self.current_state_index >= 0:
                    self.gui._show_board()
                else:
                    self.gui._show_board()
            return

        state = self.tab_states[tab_index]

        # Check dimensions
        saved_height, saved_width, saved_mines = state.get('board_dimensions', (self.height, self.width, self.num_mines))
        dimensions_changed = (saved_height != self.height or saved_width != self.width or saved_mines != self.num_mines)

        if dimensions_changed:
            self.height, self.width, self.num_mines = saved_height, saved_width, saved_mines
            self.gui.height, self.gui.width, self.gui.num_mines = self.height, self.width, self.num_mines
            self._update_flagged_array()
            self._set_game(MinesweeperGame(self.height, self.width, self.num_mines))
            self.gui._recreate_board_only(synchronous=True)
        else:
            # Ensure single source of truth
            self._set_game(self.gui.game)

        # Restore board state
        target_game = self.gui.game
        game_board = state.get('game_board', [])
        for row in range(self.height):
            for col in range(self.width):
                if row < len(game_board) and col < len(game_board[row]):
                    target_game.board[row][col] = game_board[row][col]
                else:
                    target_game.board[row][col] = 'E'

        # Restore GUI state (mutate in-place to preserve references)
        saved_flagged = state.get('flagged', [])
        if (saved_flagged and len(saved_flagged) == self.height and
            all(len(row) == self.width for row in saved_flagged)):
            for r in range(self.height):
                for c in range(self.width):
                    self.flagged[r][c] = saved_flagged[r][c]
        else:
            self._update_flagged_array()  # Reinit if mismatch

        # Flagged_indices
        self.flagged_indices.clear()
        self.flagged_indices.update(state.get('flagged_indices', []))

        # Buttons_clear
        self.buttons_clear.clear()
        self.buttons_clear.update(state.get('buttons_clear', []))
        self.moves = state.get('moves', 0)
        self.flags_placed = state.get('flags_placed', 0)

        # Restore game state
        game_state_value = state.get('game_state')
        if isinstance(game_state_value, str):
            gs_up = str(game_state_value).strip().upper()
            if gs_up in ('WON', 'DONE'):
                target_game.game_state = GameState.WON
            elif gs_up == 'LOST':
                target_game.game_state = GameState.LOST
            else:
                target_game.game_state = GameState.PROG
        else:
            target_game.game_state = GameState(game_state_value)

        target_game.cells_opened = state.get('cells_opened', 0)
        target_game.mines_triggered = state.get('mines_triggered', 0)
        self.game_started = state.get('game_started', False)

        # For non-Load Game State tabs, restore loaded state info
        if tab_index != 0:
            self.viewing_loaded_state = state.get('viewing_loaded_state', False)
            self.loaded_state_index = state.get('loaded_state_index', -1)
            self.current_state_index = state.get('current_state_index', -1)

        # Update UI
        self.gui._update_all_buttons()
        self._update_statistics()

        # Show appropriate content
        if tab_index == 0:  # Load Game State
            if self.loaded_states and self.loaded_state_index >= 0:
                self.gui._show_board()
            else:
                self.gui._show_board_info("Select a game state file from the table to load.")
        elif tab_index == 1:  # Playable Game
            if self.game_started:
                self.gui._show_board()
            else:
                self.gui._show_board_info()
        elif tab_index == 2:  # Logic Bot
            if self.bot_states and self.current_state_index >= 0:
                self.gui._show_board()
            else:
                self.gui._show_board()

        # State label is handled centrally by _update_statistics() to avoid duplicate banners.

    # Saved Game States (loaded-file replay) methods live in `gm_loaded.py`.

    # Saved Game States replay methods live in `gm_loaded.py` (LoadedReplayMixin).
    # Logic bot replay methods live in `gm_logic_bots.py` (LogicBotReplayMixin).
