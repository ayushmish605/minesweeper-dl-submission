# board_manager.py

"""
This file is my board UI manager for MinesweeperGUI.

I keep "create the grid", "update a cell", and "normalize values for display" here so
the rest of the GUI code doesn't have to think about widget layout details.
"""

from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QLayout
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QMouseEvent


class MinesweeperButton(QPushButton):
    """This is my custom cell button so right-click (flag) works cleanly in Qt."""
    
    def __init__(self, row, col, gui):
        super().__init__()
        self.row = row
        self.col = col
        self.gui = gui

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.RightButton:
            self.gui.event_handlers.handle_right_click(self.row, self.col)
            event.accept()
        else:
            super().mousePressEvent(event)


class BoardManager:
    """I use this to create/update the board widget grid."""

    @staticmethod
    def _normalize_cell_value(v):
        """
        I normalize board cell values to strings.
        Some loaded/bot states store ints (0..8) instead of strings, and the UI expects strings.
        """
        if v is None:
            return 'E'
        # Avoid treating True/False as 1/0
        if isinstance(v, bool):
            return 'E'
        if isinstance(v, (int, float)):
            return str(int(v))
        return str(v)

    def _create_board(self, gui):
        """Create the game board UI."""
        if not hasattr(gui, 'board_widget'):
            return
        
        gm = gui.game_manager  # single source of truth for board widgets

        # Create layout
        layout = QGridLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(0)
        layout.setVerticalSpacing(0)

        # Keep the board tightly packed (prevents Qt from distributing extra space
        # between cells when the scroll area/widget expands).
        layout.setSizeConstraint(QLayout.SetFixedSize)

        gm.cell_buttons.clear()
        
        # Create buttons
        for row in range(gui.height):
            for col in range(gui.width):
                button = self._create_cell_button(gui, row, col)
                layout.addWidget(button, row, col)
                gm.cell_buttons[(row, col)] = button

        gui.board_widget.setLayout(layout)

    def _create_cell_button(self, gui, row: int, col: int):
        """Create a single cell button."""
        is_dark = (row + col) % 2 == 0
        bg_color = gui.COLORS['unrevealed_dark'] if is_dark else gui.COLORS['unrevealed_light']

        button = MinesweeperButton(row, col, gui)
        button.setFixedSize(gui.cell_size, gui.cell_size)
        button.setContentsMargins(0, 0, 0, 0)

        border_width = "1px solid #424242"
        font_px = max(10, int(gui.cell_size * 0.55))
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                border: {border_width};
                font-weight: bold;
                font-size: {font_px}px;
                margin: 0px;
                padding: 0px;
            }}
        """)

        # Connect left-click handler
        button.clicked.connect(lambda _, r=row, c=col: gui.event_handlers.handle_left_click(r, c))

        return button

    def _rebuild_board_layout(self, gui):
        """Rebuild the board layout from scratch."""
        if not hasattr(gui, 'board_scroll') or not gui.board_scroll:
            return

        gm = gui.game_manager

        # Clear references first to avoid stale widget pointers
        gm.cell_buttons.clear()

        # Hard-reset the board widget to avoid Qt layout replacement weirdness
        # (this is safer than setLayout(None), and avoids warnings/leaks).
        gui.board_widget = QWidget()
        gui.board_scroll.setWidget(gui.board_widget)

        # Create new layout and buttons
        self._create_board(gui)

    def _recreate_board_only(self, gui, synchronous=False):
        """Recreate just the board UI."""
        if not hasattr(gui, 'board_widget') or not gui.board_widget:
            gui.board_widget = QWidget()
            if hasattr(gui, 'board_scroll') and gui.board_scroll:
                gui.board_scroll.setWidget(gui.board_widget)

        if synchronous:
            self._rebuild_board_layout(gui)
            self._update_all_buttons(gui)
        else:
            # Defer the recreation to avoid Qt crashes during event processing
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, lambda: self._finish_board_recreation(gui))

    def _finish_board_recreation(self, gui):
        """Finish deferred board recreation."""
        self._rebuild_board_layout(gui)
        self._update_all_buttons(gui)

    def _clear_layout(self, layout):
        """Safely clear all items from a layout."""
        if layout is None:
            return

        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                break
            widget = item.widget()
            if widget is not None:
                # Disconnect signals safely
                try:
                    widget.clicked.disconnect()
                except (TypeError, RuntimeError):
                    pass  # Signal might not be connected / already deleted

                try:
                    widget.customContextMenuRequested.disconnect()
                except (TypeError, RuntimeError):
                    pass  # Signal might not be connected / already deleted

                # Remove from parent and delete
                widget.setParent(None)
                widget.deleteLater()
    
    def _update_button(self, gui, row: int, col: int):
        """Update a single button's appearance."""
        gm = gui.game_manager
        if (row, col) not in gm.cell_buttons:
            return
        
        button = gm.cell_buttons[(row, col)]
        cell_value = self._normalize_cell_value(gui.game.board[row][col])
        value = row * gui.width + col
        
        is_dark = (row + col) % 2 == 0
        font_px = max(10, int(gui.cell_size * 0.55))

        # Preview mode: show a greyed-out board when configuring settings (Playable tab only).
        if (getattr(gm, "current_tab_index", None) == 1
            and not getattr(gm, "game_started", False)
            and getattr(gm, "preview_mode", False)
            and not getattr(gm, "viewing_loaded_state", False)):
            button.setEnabled(False)
            button.setText("")
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: #4A4A4A;
                    border: 1px solid #2F2F2F;
                    font-weight: bold;
                    font-size: {font_px}px;
                    margin: 0px;
                    padding: 0px;
                    color: #9A9A9A;
                }}
            """)
            return
        
        # Interactive / playback modes
        button.setEnabled(True)

        # Determine button state and appearance
        if value in gui.game_manager.flagged_indices:
            # Flagged
            button.setText("🚩")
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {gui.COLORS['unrevealed_dark'] if is_dark else gui.COLORS['unrevealed_light']};
                    border: 1px solid #424242;
                    font-weight: bold;
                    font-size: {font_px}px;
                    margin: 0px;
                    padding: 0px;
                    color: {gui.COLORS['flag']};
                }}
            """)
        elif cell_value == 'E':
            # Unrevealed
            button.setText("")
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {gui.COLORS['unrevealed_dark'] if is_dark else gui.COLORS['unrevealed_light']};
                    border: 1px solid #424242;
                    font-weight: bold;
                    font-size: {font_px}px;
                    margin: 0px;
                    padding: 0px;
                }}
            """)
        elif cell_value == 'M':
            # Mine
            button.setText("💣")
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {gui.COLORS['mine']};
                    border: 1px solid #424242;
                    font-weight: bold;
                    font-size: {font_px}px;
                    margin: 0px;
                    padding: 0px;
                    color: #FFFFFF;
                }}
            """)
        elif cell_value in ['0', 'B']:
            # Empty revealed (0 or B for blank)
            button.setText("")
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {gui.COLORS['revealed']};
                    border: 1px solid #424242;
                    font-weight: bold;
                    font-size: {font_px}px;
                    margin: 0px;
                    padding: 0px;
                }}
            """)
        else:
            # Number clue
            button.setText(cell_value)
            colors = {
                '1': gui.COLORS['clue_1'],
                '2': gui.COLORS['clue_2'],
                '3': gui.COLORS['clue_3'],
                '4': gui.COLORS['clue_4'],
                '5': gui.COLORS['clue_5'],
                '6': gui.COLORS['clue_6'],
                '7': gui.COLORS['clue_7'],
                '8': gui.COLORS['clue_8'],
            }
            color = colors.get(cell_value, gui.COLORS['clue_7'])
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {gui.COLORS['revealed']};
                    border: 1px solid #424242;
                    font-weight: bold;
                    font-size: {font_px}px;
                    margin: 0px;
                    padding: 0px;
                    color: {color};
                }}
            """)
    
    def _update_all_buttons(self, gui):
        """Update all buttons on the board."""
        gm = gui.game_manager
        if not gm.cell_buttons:
            return  # board not created yet

        for row in range(gui.height):
            for col in range(gui.width):
                # Keep UI updating even if a single cell contains unexpected data
                try:
                    self._update_button(gui, row, col)
                except Exception:
                    pass
