# event_handlers.py

"""
These are the event handlers for my Minesweeper GUI.

I keep them in a dedicated class so the main window stays readable and I can reason
about "what happens on click/keypress" in one place.
"""

from PyQt5.QtWidgets import QTableWidgetItem, QMessageBox, QApplication
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtCore import Qt, QTimer
from ..game import GameState
from ..board_metrics import safe_opened_from_visible, total_safe_cells
from pathlib import Path


class _SortKeyItem(QTableWidgetItem):
    """
    QTableWidgetItem that sorts using a separate key (stored in Qt.UserRole + 1).
    This lets me display pretty strings but sort by numeric timestamps/counts.
    """

    SORT_ROLE = int(Qt.UserRole) + 1

    def __init__(self, text: str, *, sort_key):
        super().__init__(str(text))
        try:
            self.setData(self.SORT_ROLE, sort_key)
        except Exception:
            pass

    def __lt__(self, other):
        try:
            return self.data(self.SORT_ROLE) < other.data(self.SORT_ROLE)
        except Exception:
            return super().__lt__(other)


class EventHandlers:
    """I use this to handle user interactions/events (clicks, table interactions, etc.)."""

    def __init__(self, gui):
        self.gui = gui
        self.game_manager = gui.game_manager
        # Saved Game States sorting defaults (Last Updated newest -> oldest).
        if not hasattr(self.gui, "_state_table_sort_col"):
            setattr(self.gui, "_state_table_sort_col", 0)
        if not hasattr(self.gui, "_state_table_sort_order"):
            setattr(self.gui, "_state_table_sort_order", int(Qt.DescendingOrder))

    def _run_with_button_loading(self, btn, *, running_text: str, fn):
        """
        Small UX helper: while a bot is running, I disable its Run button and show a loading label.
        This prevents spam-clicking and makes it obvious something is happening.
        """
        if btn is None:
            return fn()
        # If something is already running, ignore re-entrance (extra safety).
        if bool(getattr(self.game_manager, "bot_running", False)):
            return None
        try:
            old_text = btn.text()
        except Exception:
            old_text = None
        try:
            old_enabled = bool(btn.isEnabled())
        except Exception:
            old_enabled = True

        # Prevent other UI refresh paths from re-enabling buttons mid-run.
        prev_bot_running = bool(getattr(self.game_manager, "bot_running", False))
        try:
            self.game_manager.bot_running = True
        except Exception:
            pass

        def _finish():
            try:
                if old_text is not None:
                    btn.setText(old_text)
                btn.setEnabled(old_enabled)
            except Exception:
                pass
            try:
                self.game_manager.bot_running = prev_bot_running
            except Exception:
                pass

        def _run():
            try:
                fn()
            finally:
                _finish()

        # IMPORTANT: Defer the heavy work by one Qt tick so the disabled/greyed-out
        # state actually repaints before we block the UI thread.
        try:
            btn.setEnabled(False)
            if running_text:
                if old_text and "(" in str(running_text):
                    btn.setText(f"{old_text} {running_text}")
                else:
                    btn.setText(str(running_text))
        except Exception:
            pass
        try:
            QApplication.processEvents()
        except Exception:
            pass
        try:
            btn.repaint()
        except Exception:
            pass

        # IMPORTANT: use a double-queued tick. On some machines/Qt builds, a single queued call
        # can still run before the paint event that shows the disabled/text-changed state.
        try:
            QTimer.singleShot(0, lambda: QTimer.singleShot(0, _run))
        except Exception:
            # Fallback: run inline if timer isn't available for some reason.
            _run()
        return None

    def handle_left_click(self, row: int, col: int):
        """Handle left-click on a cell."""
        if not self.game_manager.game_started:
            return

        # Only allow clicks on Playable Game tab
        if self.game_manager.current_tab_index != 1:
            return

        # Don't allow clicks when viewing loaded/bot states
        if self.game_manager.viewing_loaded_state or self.game_manager.bot_states:
            return

        # Check game state.
        #
        # When "Continue after mine?" is enabled, my game logic will still mark the state as LOST
        # after a mine trigger, but I want the UI to keep accepting clicks so the run can continue.
        state = self.game_manager.game.get_game_state()
        if state != GameState.PROG:
            if not (bool(getattr(self.game_manager, "allow_mine_triggers", False)) and state == GameState.LOST):
                return

        # Check if already revealed
        if self.game_manager.game.board[row][col] != 'E':
            return

        value = row * self.game_manager.width + col
        if value in self.game_manager.flagged_indices:
            return  # Flagged

        # Process click (optionally continue after triggering a mine)
        result = self.game_manager.game.player_clicks(
            row,
            col,
            self.game_manager.buttons_clear,
            allow_mine_triggers=bool(getattr(self.game_manager, "allow_mine_triggers", False)),
        )
        self.game_manager.moves += 1

        # After first click, save game configuration for bot replay
        if self.game_manager.game._board_initialized and self.game_manager.original_actual_board is None:
            actual_board = self.game_manager.game.get_actual_board()
            if actual_board:
                self.game_manager.original_actual_board = [row[:] for row in actual_board]
                self.game_manager.original_first_click = (row, col)

                # Enable Logic Bot tab
                if hasattr(self.gui, 'board_tabs'):
                    # Enable outer Bots tab (contains Logic/NN sub-tabs)
                    self.gui.board_tabs.setTabEnabled(2, True)
                    # Run buttons are gated by difficulty + availability.
                    # I let the per-tab `prepare_*` methods decide what is enabled.
                    try:
                        self.game_manager.prepare_logic_bot_tab()
                        self.game_manager.prepare_nn_mine_tab()
                        self.game_manager.prepare_nn_move_tab()
                        self.game_manager.prepare_nn_think_tab()
                    except Exception:
                        pass

        # Update button and clear revealed cells
        self.gui._update_button(row, col)
        for button_index in self.game_manager.buttons_clear:
            r = button_index // self.game_manager.width
            c = button_index % self.game_manager.width
            self.gui._update_button(r, c)
        self.game_manager.buttons_clear.clear()

        # Safety: ensure the UI never desyncs from the underlying board.
        # This is cheap for our board sizes and prevents "some cells didn't repaint"
        # if the game logic revealed cells not included in buttons_clear for any reason.
        self.gui._update_all_buttons()

        # Handle game end
        if result == "Lost" and not bool(getattr(self.game_manager, "allow_mine_triggers", False)):
            self._handle_game_lost(row, col)
        elif result == "Win":
            self._handle_game_win()
        else:
            self.game_manager._update_statistics()
            if result == "Lost" and bool(getattr(self.game_manager, "allow_mine_triggers", False)):
                # Mine triggered but continuing
                if hasattr(self.gui, "state_label"):
                    self.gui.state_label.setText("Mine Triggered (continuing)")
                    try:
                        self.gui.state_label.setStyleSheet(self.game_manager._get_loss_label_style())
                    except Exception:
                        pass

        # Collect training states (after processing the click)
        if getattr(self.gui, "collect_states", False) and getattr(self.gui, "state_collector", None):
            try:
                self.gui.state_collector.capture_state(self.game_manager.game, action=(row, col))
                # Persist correct flags (flags on mines) ONLY while the game is in progress.
                # Win/loss handlers may clear flags for display; we do not want that to overwrite
                # flags_on_mines, since deleting the terminal step should restore previous flags.
                try:
                    if self.game_manager.game.get_game_state() == GameState.PROG:
                        self.gui.state_collector.update_flags_on_mines(self.game_manager.game, self.game_manager.flagged)
                except Exception:
                    pass
                # If this was the first captured state, the file will have just been created.
                # Refresh the Load Game States table so it appears immediately.
                try:
                    if hasattr(self.gui, "state_table") and len(self.gui.state_collector.states) == 1:
                        self._populate_state_table()
                except Exception:
                    pass
            except Exception:
                # Never break gameplay due to data collection
                pass

    def handle_right_click(self, row: int, col: int):
        """Handle right-click on a cell (flag/unflag)."""
        if not self.game_manager.game_started:
            return

        # Only allow clicks on Playable Game tab
        if self.game_manager.current_tab_index != 1:
            return

        # Don't allow flags when viewing loaded/bot states
        if self.game_manager.viewing_loaded_state or self.game_manager.bot_states:
            return

        state = self.game_manager.game.get_game_state()
        if state != GameState.PROG:
            if not (bool(getattr(self.game_manager, "allow_mine_triggers", False)) and state == GameState.LOST):
                return

        # Allow flagging/unflagging only unrevealed cells
        if not self.game_manager.flagged[row][col] and self.game_manager.game.board[row][col] != 'E':
            return

        value = row * self.game_manager.width + col

        if self.game_manager.flagged[row][col]:
            # Unflag
            self.game_manager.flagged[row][col] = False
            self.game_manager.flags_placed -= 1
            self.game_manager.flagged_indices.discard(value)
        else:
            # Flag - only if cell is unrevealed
            if self.game_manager.game.board[row][col] == 'E':
                self.game_manager.flagged[row][col] = True
                self.game_manager.flags_placed += 1
                self.game_manager.flagged_indices.add(value)

        self.gui._update_button(row, col)
        self.game_manager._update_statistics()

        # Persist correct flags (flags placed on mines) to the current game-state file.
        if getattr(self.gui, "collect_states", False) and getattr(self.gui, "state_collector", None):
            try:
                self.gui.state_collector.update_flags_on_mines(self.game_manager.game, self.game_manager.flagged)
                # Refresh the Load Game States table so the flags count is visible immediately.
                if hasattr(self.gui, "state_table"):
                    self._populate_state_table()
            except Exception:
                pass

    def _handle_game_lost(self, clicked_row: int, clicked_col: int):
        """Handle game lost state."""
        # Show all mines
        actual_board = self.game_manager.game.get_actual_board()
        if actual_board:
            for row in range(self.game_manager.height):
                for col in range(self.game_manager.width):
                    if actual_board[row][col] == "M":
                        self.game_manager.game.board[row][col] = "M"
                        value = row * self.game_manager.width + col
                        # Remove from flagged indices and clear flagged state if it was flagged
                        if value in self.game_manager.flagged_indices:
                            self.game_manager.flagged_indices.discard(value)
                            self.game_manager.flagged[row][col] = False
                            self.game_manager.flags_placed -= 1
                        self.gui._update_button(row, col)

        self.game_manager._update_statistics()
        if hasattr(self.gui, "state_details"):
            self.gui.state_details.setText(
                "Result: LOSS\n\n"
                "Next steps:\n"
                "- Click 'New Game' to start a fresh board."
            )

    def _handle_game_win(self):
        """Handle game won state."""
        # Reveal all remaining cells
        for button_index in self.game_manager.buttons_clear:
            r = button_index // self.game_manager.width
            c = button_index % self.game_manager.width
            self.gui._update_button(r, c)
        self.game_manager.buttons_clear.clear()

        self.game_manager._update_statistics()
        if hasattr(self.gui, "state_details"):
            self.gui.state_details.setText(
                "Result: WIN\n\n"
                "Next steps:\n"
                "- Click 'New Game' to play again."
            )

    def handle_key_press(self, event):
        """Handle keyboard events."""
        # Handle arrow keys based on current tab
        tab_index = self.game_manager.current_tab_index

        if tab_index == 0:  # Load Game State tab
            if self.game_manager.loaded_states:
                if event.key() == Qt.Key_Right:
                    self.game_manager.navigate_loaded_state(1)
                elif event.key() == Qt.Key_Left:
                    self.game_manager.navigate_loaded_state(-1)
                else:
                    event.ignore()
                    return
            else:
                event.ignore()
                return

        elif tab_index == 2:  # Bots container tab
            # Delegate arrow-key navigation based on the selected bot sub-tab.
            try:
                sub = int(self.gui.bots_tabs.currentIndex()) if hasattr(self.gui, "bots_tabs") else 0
            except Exception:
                sub = 0
            if event.key() not in (Qt.Key_Right, Qt.Key_Left):
                event.ignore()
                return
            direction = 1 if event.key() == Qt.Key_Right else -1
            if sub == 0:  # Logic Bot
                self.game_manager.navigate_bot_state(direction)
            elif sub == 1:  # NN Mine Prediction
                self.game_manager.navigate_nn_mine_state(direction)
            elif sub == 2:  # NN Move Prediction
                self.game_manager.navigate_nn_move_state(direction)
            elif sub == 3:  # NN Thinking Deeper
                self.game_manager.navigate_nn_think_state(direction)
            else:
                event.ignore()
                return

        else:
            event.ignore()
            return

        event.accept()

    def handle_tab_change(self, index: int):
        """Handle tab change."""
        # Save current tab state
        if hasattr(self.game_manager, 'current_tab_index'):
            self.game_manager.save_tab_state(self.game_manager.current_tab_index)

        # IMPORTANT: update current_tab_index BEFORE running any tab-specific logic so that:
        # - selecting a row in Saved Game States triggers "selection == click" behavior
        # - entering the Bots tab actually runs the prepare_* methods the first time
        self.game_manager.current_tab_index = index

        # Handle tab-specific logic
        if index == 0:  # Load Game State tab
            if hasattr(self.gui, 'state_table'):
                self._populate_state_table()
                # Auto-select the most recent file (top row) when entering the tab.
                try:
                    if self.gui.state_table.rowCount() > 0:
                        self.gui.state_table.setCurrentCell(0, 0)
                        self.gui.state_table.selectRow(0)
                except Exception:
                    pass
        elif index == 1:  # Playable Game tab
            self.game_manager.viewing_loaded_state = False
        elif index == 2:  # Bots container tab
            # Prepare whichever bot sub-tab is currently selected.
            self.handle_bot_subtab_change(getattr(self.gui, "bots_tabs", None).currentIndex() if hasattr(self.gui, "bots_tabs") else 0)

        # Restore target tab state
        self.game_manager.restore_tab_state(index)
        self.game_manager.current_tab_index = index

    def handle_bot_jump_input(self):
        """Handle jump to bot action number."""
        if getattr(self.game_manager, "current_tab_index", None) != 2:
            return
        # Only when the Logic Bot sub-tab is active
        if hasattr(self.gui, "bots_tabs") and int(self.gui.bots_tabs.currentIndex()) != 0:
            return
        if hasattr(self.gui, "bot_jump_input"):
            self.game_manager.jump_to_bot_state(self.gui.bot_jump_input.text())

    def handle_nn_mine_jump_input(self):
        if getattr(self.game_manager, "current_tab_index", None) != 2:
            return
        if hasattr(self.gui, "bots_tabs") and int(self.gui.bots_tabs.currentIndex()) != 1:
            return
        if hasattr(self.gui, "nn_mine_jump_input"):
            self.game_manager.jump_to_nn_mine_state(self.gui.nn_mine_jump_input.text())

    def handle_nn_move_jump_input(self):
        if getattr(self.game_manager, "current_tab_index", None) != 2:
            return
        if hasattr(self.gui, "bots_tabs") and int(self.gui.bots_tabs.currentIndex()) != 2:
            return
        if hasattr(self.gui, "nn_move_jump_input"):
            self.game_manager.jump_to_nn_move_state(self.gui.nn_move_jump_input.text())

    def handle_nn_think_jump_input(self):
        if getattr(self.game_manager, "current_tab_index", None) != 2:
            return
        if hasattr(self.gui, "bots_tabs") and int(self.gui.bots_tabs.currentIndex()) != 3:
            return
        if hasattr(self.gui, "nn_think_jump_input"):
            self.game_manager.jump_to_nn_think_state(self.gui.nn_think_jump_input.text())

    def handle_bot_subtab_change(self, sub_index: int):
        """
        Called when the user switches bot sub-tabs inside the outer Bots tab.
        """
        try:
            sub = int(sub_index)
        except Exception:
            sub = 0
        # Only do work when Bots tab is active
        if getattr(self.game_manager, "current_tab_index", None) != 2:
            return
        try:
            if sub == 0:
                self.game_manager.prepare_logic_bot_tab()
            elif sub == 1:
                self.game_manager.prepare_nn_mine_tab()
            elif sub == 2:
                self.game_manager.prepare_nn_move_tab()
            elif sub == 3:
                self.game_manager.prepare_nn_think_tab()
        except Exception:
            pass

    def handle_run_nn_mine_bot(self):
        """Run the Task 1 NN bot (mine prediction)."""
        def _work():
            self.game_manager.run_nn_mine_demo()
            # Export NN run (if collection is enabled), same idea as LogicBot export.
            if getattr(self.gui, "collect_states", False) and getattr(self.gui, "state_collector", None):
                try:
                    gm = self.game_manager
                    if gm.original_actual_board and gm.nn_mine_states:
                        mine_ct = sum(1 for rr in gm.original_actual_board for v in rr if v == "M")
                        total_safe = total_safe_cells(height=int(gm.height), width=int(gm.width), num_mines=int(mine_ct))
                        states = []
                        final_state = GameState.PROG.value
                        mines_triggered_last = 0
                        for s in gm.nn_mine_states:
                            if not isinstance(s, dict):
                                continue
                            board = s.get("board", []) or []
                            mines_triggered = int(s.get("mines_triggered", 0) or 0)
                            states.append({"action": s.get("action"), "board": board, "mines_triggered": mines_triggered})
                            mines_triggered_last = int(mines_triggered)
                            if safe_opened_from_visible(board) >= int(total_safe):
                                final_state = GameState.DONE.value if mines_triggered > 0 else GameState.WON.value
                                break
                        # If we didn't finish, match LogicBot file semantics: LOST if any mine was triggered.
                        if final_state == GameState.PROG.value and mines_triggered_last > 0:
                            final_state = GameState.LOST.value
                        path = self.gui.state_collector.export_precomputed_states(
                            mode="nn_task1",
                            height=gm.height,
                            width=gm.width,
                            num_mines=mine_ct,
                            actual_board_initial=gm.original_actual_board,
                            states=states,
                            flags_on_mines=[],
                            final_game_state=str(final_state),
                        )
                        if path and hasattr(self, "_populate_state_table"):
                            self._populate_state_table()
                except Exception:
                    pass

        self._run_with_button_loading(getattr(self.gui, "run_nn_mine_btn", None), running_text="(Loading...)", fn=_work)

    def handle_run_nn_move_bot(self):
        """Run the Task 2 NN bot (move prediction / actor-critic)."""
        def _work():
            self.game_manager.run_nn_move_demo()
            if getattr(self.gui, "collect_states", False) and getattr(self.gui, "state_collector", None):
                try:
                    gm = self.game_manager
                    if gm.original_actual_board and gm.nn_move_states:
                        mine_ct = sum(1 for rr in gm.original_actual_board for v in rr if v == "M")
                        total_safe = total_safe_cells(height=int(gm.height), width=int(gm.width), num_mines=int(mine_ct))
                        states = []
                        final_state = GameState.PROG.value
                        mines_triggered_last = 0
                        for s in gm.nn_move_states:
                            if not isinstance(s, dict):
                                continue
                            board = s.get("board", []) or []
                            mines_triggered = int(s.get("mines_triggered", 0) or 0)
                            states.append({"action": s.get("action"), "board": board, "mines_triggered": mines_triggered})
                            mines_triggered_last = int(mines_triggered)
                            if safe_opened_from_visible(board) >= int(total_safe):
                                final_state = GameState.DONE.value if mines_triggered > 0 else GameState.WON.value
                                break
                        if final_state == GameState.PROG.value and mines_triggered_last > 0:
                            final_state = GameState.LOST.value
                        path = self.gui.state_collector.export_precomputed_states(
                            mode="nn_task2",
                            height=gm.height,
                            width=gm.width,
                            num_mines=mine_ct,
                            actual_board_initial=gm.original_actual_board,
                            states=states,
                            flags_on_mines=[],
                            final_game_state=str(final_state),
                        )
                        if path and hasattr(self, "_populate_state_table"):
                            self._populate_state_table()
                except Exception:
                    pass

        self._run_with_button_loading(getattr(self.gui, "run_nn_move_btn", None), running_text="(Loading...)", fn=_work)

    def handle_run_nn_think_bot(self):
        """Run the Task 3 NN bot (thinking deeper)."""
        def _work():
            self.game_manager.run_nn_think_demo()
            if getattr(self.gui, "collect_states", False) and getattr(self.gui, "state_collector", None):
                try:
                    gm = self.game_manager
                    if gm.original_actual_board and gm.nn_think_states:
                        mine_ct = sum(1 for rr in gm.original_actual_board for v in rr if v == "M")
                        total_safe = total_safe_cells(height=int(gm.height), width=int(gm.width), num_mines=int(mine_ct))
                        states = []
                        final_state = GameState.PROG.value
                        mines_triggered_last = 0
                        for s in gm.nn_think_states:
                            if not isinstance(s, dict):
                                continue
                            board = s.get("board", []) or []
                            mines_triggered = int(s.get("mines_triggered", 0) or 0)
                            states.append({"action": s.get("action"), "board": board, "mines_triggered": mines_triggered})
                            mines_triggered_last = int(mines_triggered)
                            if safe_opened_from_visible(board) >= int(total_safe):
                                final_state = GameState.DONE.value if mines_triggered > 0 else GameState.WON.value
                                break
                        if final_state == GameState.PROG.value and mines_triggered_last > 0:
                            final_state = GameState.LOST.value
                        path = self.gui.state_collector.export_precomputed_states(
                            mode="nn_task3",
                            height=gm.height,
                            width=gm.width,
                            num_mines=mine_ct,
                            actual_board_initial=gm.original_actual_board,
                            states=states,
                            flags_on_mines=[],
                            final_game_state=str(final_state),
                        )
                        if path and hasattr(self, "_populate_state_table"):
                            self._populate_state_table()
                except Exception:
                    pass

        self._run_with_button_loading(getattr(self.gui, "run_nn_think_btn", None), running_text="(Loading...)", fn=_work)

    def handle_state_table_click(self, index):
        """Handle click on state table."""
        try:
            row = index.row()
        except Exception:
            return
        self._activate_state_table_row(row)

    def handle_state_table_selection_changed(self):
        """
        When navigating the Load Game States table with keyboard, treat selection
        like a click (load the selected file).
        """
        if getattr(self.game_manager, "current_tab_index", None) != 0:
            return
        if not hasattr(self.gui, "state_table") or self.gui.state_table is None:
            return

        model = self.gui.state_table.selectionModel()
        if model is None:
            return

        selected_rows = model.selectedRows()
        if not selected_rows:
            return

        self._activate_state_table_row(selected_rows[0].row())

    def handle_state_table_header_clicked(self, column: int):
        """
        Saved Game States sorting: click header to toggle ascending/descending.

        Not every column is sortable (categorical columns like Mode/Status are not).
        """
        if not hasattr(self.gui, "state_table") or self.gui.state_table is None:
            return

        # Sortable columns:
        # 0 Last Updated (time)
        # 3 Mines Triggered (numeric)
        # 4 Size (numeric-ish)
        # 5 States (numeric)
        # 6 File (string)
        sortable = {0, 3, 4, 5, 6}
        if int(column) not in sortable:
            # Keep indicator unchanged.
            try:
                header = self.gui.state_table.horizontalHeader()
                col = int(getattr(self.gui, "_state_table_sort_col", 0))
                order = Qt.SortOrder(int(getattr(self.gui, "_state_table_sort_order", int(Qt.DescendingOrder))))
                header.setSortIndicator(col, order)
            except Exception:
                pass
            return

        prev_col = int(getattr(self.gui, "_state_table_sort_col", 0))
        prev_order_int = int(getattr(self.gui, "_state_table_sort_order", int(Qt.DescendingOrder)))

        if int(column) == int(prev_col):
            new_order = Qt.AscendingOrder if prev_order_int == int(Qt.DescendingOrder) else Qt.DescendingOrder
        else:
            # New column: default direction depends on column.
            new_order = Qt.DescendingOrder if int(column) == 0 else Qt.AscendingOrder

        try:
            self.gui._state_table_sort_col = int(column)
            self.gui._state_table_sort_order = int(new_order)
        except Exception:
            pass

        try:
            self.gui.state_table.setSortingEnabled(True)
            self.gui.state_table.sortItems(int(column), new_order)
            self.gui.state_table.setSortingEnabled(False)
        except Exception:
            pass
        try:
            header = self.gui.state_table.horizontalHeader()
            header.setSortIndicator(int(column), new_order)
        except Exception:
            pass

    def _activate_state_table_row(self, row: int):
        """Load the game state file corresponding to a table row (if present)."""
        if not hasattr(self.gui, "state_table") or self.gui.state_table is None:
            return
        if row < 0 or row >= self.gui.state_table.rowCount():
            return

        # We store filepath on the timestamp item (col 0).
        timestamp_item = self.gui.state_table.item(row, 0)
        if timestamp_item is None:
            return

        filepath = timestamp_item.data(Qt.UserRole)
        if not filepath:
            return

        self.game_manager.load_game_state_file(filepath)

    def handle_jump_input(self):
        """Handle jump to action number."""
        if hasattr(self.gui, 'state_jump_input') and self.game_manager.current_tab_index == 0:
            action_num = self.gui.state_jump_input.text()
            self.game_manager.jump_to_state(action_num)

    def handle_continue_playing_from_loaded(self):
        """Continue playing the currently viewed loaded snapshot (if allowed)."""
        # Manual-only restriction
        try:
            if str((self.game_manager.loaded_metadata or {}).get("mode") or "") != "manual":
                QMessageBox.information(self.gui, "Manual Only", "Continue Playing is only available for manual games.")
                return
        except Exception:
            pass
        ok = self.game_manager.continue_playing_from_loaded_state()
        if not ok:
            QMessageBox.information(
                self.gui,
                "Cannot Continue",
                "Can't continue from this snapshot.\n\n"
                "Requirements:\n"
                "- The loaded file must be PROG (or LOST if 'Continue after mine?' is enabled)\n"
                "- The current selected action must be PROG (or LOST if 'Continue after mine?' is enabled)\n"
                "- The file must include metadata['actual_board_initial']"
            )

    def handle_run_logic_bot(self):
        """Run the Logic Bot replay for the currently loaded playable game."""
        def _work():
            self.game_manager.run_bot_demo()
            # Keep Run enabled; LogicBot can take random fallback moves, so reruns may differ.
            # Export a logic-bot states file for this unique board (if collection is enabled).
            if getattr(self.gui, "collect_states", False) and getattr(self.gui, "state_collector", None):
                try:
                    gm = self.game_manager
                    if gm.original_actual_board and gm.bot_states:
                        mine_ct = sum(1 for rr in gm.original_actual_board for v in rr if v == "M")
                        states = []
                        for s in gm.bot_states:
                            if not isinstance(s, dict):
                                continue
                            board = s.get("board", [])
                            action = s.get("action", None)
                            mines_triggered = s.get("mines_triggered", 0)
                            states.append(
                                {
                                    "action": list(action) if isinstance(action, (list, tuple)) else action,
                                    "board": board,
                                    "mines_triggered": int(mines_triggered) if mines_triggered is not None else 0,
                                }
                            )
                        path = self.gui.state_collector.export_precomputed_states(
                            mode="logic",
                            height=gm.height,
                            width=gm.width,
                            num_mines=mine_ct,
                            actual_board_initial=gm.original_actual_board,
                            states=states,
                            flags_on_mines=getattr(gm, "bot_flags_on_mines", []) or [],
                            final_game_state=str((gm.bot_states[-1] or {}).get("game_state") or GameState.PROG.value),
                        )
                        if path and hasattr(self, "_populate_state_table"):
                            self._populate_state_table()
                except Exception:
                    pass
            if hasattr(self.gui, "board_tabs"):
                self.gui.board_tabs.setTabEnabled(2, True)
                self.gui.board_tabs.setCurrentIndex(2)
                self.game_manager.current_tab_index = 2

        self._run_with_button_loading(getattr(self.gui, "run_logic_bot_btn", None), running_text="(Loading...)", fn=_work)

    def handle_delete_last_action(self):
        """Delete the last action in the currently loaded file (only at the last action)."""
        if not self.game_manager.loaded_states or self.game_manager.loaded_state_index < 0:
            return
        # Manual-only restriction
        try:
            if str((self.game_manager.loaded_metadata or {}).get("mode") or "") != "manual":
                QMessageBox.information(self.gui, "Manual Only", "Delete Last Action is only available for manual games.")
                return
        except Exception:
            pass
        if self.game_manager.loaded_state_index != len(self.game_manager.loaded_states) - 1:
            QMessageBox.information(self.gui, "Not Last Action", "You can only delete the last action in the file.")
            return

        resp = QMessageBox.question(
            self.gui,
            "Delete Last Action?",
            "This will permanently delete the last saved action from the game state file.\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if resp != QMessageBox.Yes:
            return

        ok = self.game_manager.delete_last_loaded_action()
        if not ok:
            QMessageBox.warning(self.gui, "Delete Failed", "Could not delete the last action (file may be read-only or invalid).")

    def handle_delete_loaded_file(self):
        """Delete the currently loaded game-states file from disk."""
        if not getattr(self.game_manager, "loaded_filepath", None):
            return

        resp = QMessageBox.question(
            self.gui,
            "Delete File?",
            "This will permanently delete the selected game-states JSON file from disk.\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if resp != QMessageBox.Yes:
            return

        ok = self.game_manager.delete_loaded_file()
        if not ok:
            QMessageBox.warning(self.gui, "Delete Failed", "Could not delete the file (it may be read-only or missing).")
            return

        # Refresh list
        try:
            self._populate_state_table()
        except Exception:
            pass

    def handle_new_game(self):
        """Handle new game button."""
        self.game_manager.reset_game_state(start_game=True)

        # Start a new state-collection file (if enabled)
        if getattr(self.gui, "collect_states", False) and getattr(self.gui, "state_collector", None):
            try:
                self.gui.state_collector.start_game(
                    mode="manual",
                    height=self.game_manager.height,
                    width=self.game_manager.width,
                    num_mines=self.game_manager.num_mines,
                    seed=None,
                    actual_board=None,
                )
            except Exception:
                pass

            # Refresh the Load Game State table so new files show up immediately
            try:
                if hasattr(self.gui, "state_table"):
                    self._populate_state_table()
            except Exception:
                pass

        # Update status area
        if hasattr(self.gui, "state_label"):
            self.gui.state_label.setText("Game State: In Progress")
            # Always set a style when setting the label text
            try:
                self.gui.state_label.setStyleSheet(self.game_manager._get_state_label_style())
            except Exception:
                pass
        if hasattr(self.gui, "state_details"):
            self.gui.state_details.setText(
                "Game started.\n\n"
                "Controls:\n"
                "- Left-click: reveal a cell\n"
                "- Right-click: place/remove a flag\n"
                "- \u2190 \u2192: navigate loaded/bot replays"
            )

    def handle_settings(self):
        """Handle settings dialog."""
        from .dialogs import GameSettingsDialog

        dialog = GameSettingsDialog(
            self.game_manager.height,
            self.game_manager.width,
            self.game_manager.num_mines,
            self.game_manager.cell_size,
            self.gui
        )

        if dialog.exec_() == dialog.Accepted:
            settings = dialog.get_settings()
            # Defer settings application to avoid Qt conflicts
            self.gui._pending_settings = settings
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, self.gui._apply_pending_settings)

    def _populate_state_table(self):
        """Populate the state table with available files."""
        if not hasattr(self.gui, 'state_table') or not self.gui.state_collector:
            return

        from pathlib import Path
        import json
        import re
        from datetime import datetime

        output_dir = Path(self.gui.state_collector.output_dir)
        game_files = sorted(output_dir.glob("game_*.json"), reverse=True)

        # Prevent reordering while inserting rows.
        try:
            self.gui.state_table.setSortingEnabled(False)
        except Exception:
            pass

        self.gui.state_table.setRowCount(len(game_files))

        def _parse_ts(ts_value, file_path: Path):
            """
            Parse common timestamp formats into a datetime (or None).
            Supports:
            - "YYYYMMDD_HHMMSS" (common in filenames / older metadata)
            - ISO-ish strings
            - epoch seconds (int/float)
            Falls back to file mtime if metadata is missing/unknown.
            """
            dt = None

            if isinstance(ts_value, (int, float)):
                try:
                    dt = datetime.fromtimestamp(ts_value)
                except Exception:
                    dt = None

            if dt is None and isinstance(ts_value, str):
                s = ts_value.strip()
                if s and s.lower() not in {"unknown", "none", "null"}:
                    # YYYYMMDD_HHMMSS
                    m = re.match(r"^(?P<d>\d{8})[_-](?P<t>\d{6})$", s)
                    if m:
                        try:
                            dt = datetime.strptime(f"{m.group('d')}_{m.group('t')}", "%Y%m%d_%H%M%S")
                        except Exception:
                            dt = None
                    else:
                        # Try ISO format
                        try:
                            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                        except Exception:
                            dt = None

            if dt is None:
                # Try parsing from filename: game_YYYYMMDD_HHMMSS_*.json
                m = re.search(r"game_(\d{8})_(\d{6})", file_path.name)
                if m:
                    try:
                        dt = datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")
                    except Exception:
                        dt = None

            if dt is None:
                try:
                    dt = datetime.fromtimestamp(file_path.stat().st_mtime)
                except Exception:
                    dt = None

            return dt

        def _human_dt(dt, fallback) -> str:
            if dt is None:
                return str(fallback) if fallback is not None else "Unknown"
            # Human-readable, but not overly long.
            return dt.strftime("%b %d, %Y %I:%M:%S %p").lstrip("0").replace(" 0", " ")

        def _derive_status(meta: dict, data: dict) -> str:
            """
            File-level metadata["game_state"] is stored as one of:
              WON / LOST / DONE / PROG
            We rely on that directly (no fallback).
            """
            try:
                gs = meta.get("game_state") if isinstance(meta, dict) else None
                if isinstance(gs, str):
                    gs_up = gs.strip().upper()
                    if gs_up in ("WON", "LOST", "DONE", "PROG"):
                        return gs_up
            except Exception:
                pass
            return "PROG"

        def _derive_mines_triggered(meta: dict, data: dict) -> int:
            """
            mines_triggered is stored explicitly at the file level in metadata["mines_triggered"].
            No fallback/derivation.
            """
            try:
                if isinstance(meta, dict):
                    return int(meta.get("mines_triggered", 0) or 0)
            except Exception:
                pass
            return 0

        def _derive_total_mines(meta: dict, data: dict) -> str:
            """
            Total mine count for display (prefers metadata.num_mines; falls back to counting
            mines in metadata.actual_board_initial when present).
            """
            try:
                if isinstance(meta, dict) and meta.get("num_mines") is not None:
                    return str(int(meta.get("num_mines")))
            except Exception:
                pass
            try:
                actual = None
                if isinstance(meta, dict):
                    actual = meta.get("actual_board_initial")
                if isinstance(actual, list):
                    mine_ct = sum(1 for rr in actual for v in rr if v == "M")
                    return str(int(mine_ct))
            except Exception:
                pass
            return "?"

        for row, filepath in enumerate(game_files):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                metadata = data.get('metadata', {})
                # Display "last updated" (preferred), not creation time.
                ts_raw = (
                    metadata.get("last_updated")
                    or data.get("last_updated")
                    or metadata.get("created_at")
                    or metadata.get("timestamp")
                    or "Unknown"
                )
                dt = _parse_ts(ts_raw, filepath)
                timestamp = _human_dt(dt, ts_raw)
                mode = metadata.get('mode', 'Unknown')
                # Human-friendly mode names (centralized in bot_tab_registry).
                try:
                    from .bot_tab_registry import mode_display_name
                    mode_text = mode_display_name(str(mode))
                except Exception:
                    mode_text = str(mode).capitalize() if isinstance(mode, str) else str(mode)
                status = _derive_status(metadata, data)
                height = metadata.get('height', '?')
                width = metadata.get('width', '?')
                mines_triggered = _derive_mines_triggered(metadata, data)
                total_mines = _derive_total_mines(metadata, data)
                total_states = data.get('total_states', len(data.get('states', [])))
                filename = filepath.name

                # Create table items
                # Sort key for Last Updated: epoch seconds.
                try:
                    ts_sort = float(dt.timestamp()) if dt is not None else float(filepath.stat().st_mtime)
                except Exception:
                    ts_sort = 0.0
                timestamp_item = _SortKeyItem(timestamp, sort_key=ts_sort)
                mode_item = QTableWidgetItem(str(mode_text))
                # Status formatting
                status_norm = str(status).upper()
                if status_norm == "WON":
                    status_text = "WON"
                    status_color = QColor("#4CAF50")
                elif status_norm == "LOST":
                    status_text = "LOST"
                    status_color = QColor("#D32F2F")
                elif status_norm == "DONE":
                    status_text = "DONE"
                    status_color = QColor("#2196F3")
                else:
                    status_text = "PROG"
                    status_color = QColor("#FF9800")

                status_item = QTableWidgetItem(status_text)
                status_item.setForeground(QBrush(status_color))

                mines_trig_item = _SortKeyItem(f"{int(mines_triggered)}/{total_mines}", sort_key=int(mines_triggered))
                try:
                    hs = int(height)
                    ws = int(width)
                    size_sort = int(hs) * int(ws)
                except Exception:
                    size_sort = 0
                size_item = _SortKeyItem(f"{height}x{width}", sort_key=size_sort)
                try:
                    states_sort = int(total_states)
                except Exception:
                    states_sort = 0
                states_item = _SortKeyItem(str(total_states), sort_key=states_sort)
                filename_item = _SortKeyItem(filename, sort_key=str(filename))

                # Style mode column
                mode_id = str(mode)
                if mode_id == 'manual':
                    mode_item.setForeground(QBrush(QColor('#4CAF50')))  # Green
                elif mode_id in ('logic', 'nn_task1', 'nn_task2', 'nn_task3', 'bot'):
                    mode_item.setForeground(QBrush(QColor('#2196F3')))  # Blue
                elif mode_id == 'demo':
                    mode_item.setForeground(QBrush(QColor('#FF9800')))  # Orange

                # Store filepath
                timestamp_item.setData(Qt.UserRole, str(filepath))

                self.gui.state_table.setItem(row, 0, timestamp_item)
                self.gui.state_table.setItem(row, 1, mode_item)
                self.gui.state_table.setItem(row, 2, status_item)
                self.gui.state_table.setItem(row, 3, mines_trig_item)
                self.gui.state_table.setItem(row, 4, size_item)
                self.gui.state_table.setItem(row, 5, states_item)
                self.gui.state_table.setItem(row, 6, filename_item)

            except Exception as e:
                error_item = QTableWidgetItem(f"Error: {str(e)}")
                self.gui.state_table.setItem(row, 0, error_item)

        # Do NOT call resizeColumnsToContents() here: it will override explicit column widths
        # and can truncate important headers (e.g., "Mines Triggered").
        # Column order (see main_window.py):
        # 0 Last Updated | 1 Mode | 2 Status | 3 Mines Triggered | 4 Size | 5 States | 6 File
        self.gui.state_table.setColumnWidth(0, 170)  # Last Updated (shrink)
        self.gui.state_table.setColumnWidth(1, 150)  # Mode (fits "NN Task 1/2/3")
        self.gui.state_table.setColumnWidth(2, 80)   # Status
        self.gui.state_table.setColumnWidth(3, 140)  # Mines Triggered (needs space for header)
        self.gui.state_table.setColumnWidth(4, 70)   # Size
        self.gui.state_table.setColumnWidth(5, 70)   # States
        # Column 6 is File and should remain stretch-managed by the header.

        # Apply current sort (default: Last Updated desc).
        try:
            sort_col = int(getattr(self.gui, "_state_table_sort_col", 0))
            sort_order = Qt.SortOrder(int(getattr(self.gui, "_state_table_sort_order", int(Qt.DescendingOrder))))
            self.gui.state_table.setSortingEnabled(True)
            self.gui.state_table.sortItems(sort_col, sort_order)
            self.gui.state_table.setSortingEnabled(False)
            try:
                header = self.gui.state_table.horizontalHeader()
                header.setSortIndicator(sort_col, sort_order)
            except Exception:
                pass
        except Exception:
            pass
