# Define the main PyQt window here (main_window.py).

"""
This is my main PyQt window for Minesweeper.

I use composition (manager/helper objects) instead of a deep inheritance tree because it
keeps the GUI easier to debug and easier to extend with new bot tabs.
"""

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QFrame, QScrollArea, QSplitter,
                             QStackedWidget, QTabWidget, QTableWidget, QTableWidgetItem,
                             QAbstractItemView, QLineEdit, QShortcut, QButtonGroup, QCheckBox, QSizePolicy, QHeaderView, QComboBox)
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QColor, QKeySequence, QIntValidator
from typing import Dict

from ..game import MinesweeperGame
from .components import StyledButton
from .constants import PRESETS
from .delegates import PreserveForegroundOnSelectDelegate
from .game_manager import GameManager
from .event_handlers import EventHandlers
from .board_manager import BoardManager


class MinesweeperGUI(QMainWindow):
    """
    This is my main GUI class: it handles settings, gameplay, bot demos, and loading/replaying saved states.
    """
    
    # Constrain cell sizes so the board stays usable.
    CELL_MIN = 18
    CELL_MAX = 34
    
    # Keep a single color scheme dict so styling stays consistent.
    COLORS = {
        'unrevealed_light': '#B2DFDB',
        'unrevealed_dark': '#80CBC4',
        'revealed': '#F5F5DC',
        'mine': '#D32F2F',
        'flag': '#FF5252',
        'clue_1': '#1976D2',
        'clue_2': '#388E3C',
        'clue_3': '#D32F2F',
        'clue_4': '#7B1FA2',
        'clue_5': '#F57C00',
        'clue_6': '#0288D1',
        'clue_7': '#000000',
        'clue_8': '#616161',
        'background': '#1E1E1E',
        'text': '#FFFFFF',
        'text_secondary': '#B0B0B0',
        'input_bg': '#2E2E2E',
        'input_border': '#555555',
        'border': '#424242',
        'header_bg': '#2E2E2E',
        'row_alt': '#2A2A2A',
        'selection': '#80CBC4',
    }
    
    def __init__(self, collect_states: bool = False, state_collector=None):
        super().__init__()
        
        # Set default game parameters.
        self.height = 22
        self.width = 22
        self.num_mines = 80
        self.cell_size = 25
        self.preset_name = self._infer_preset_name()
        
        # Wire optional state collection here.
        self.collect_states = collect_states
        self.state_collector = state_collector
        
        # Initialize the game and my managers here.
        self.game = MinesweeperGame(self.height, self.width, self.num_mines)
        self.game_manager = GameManager(self)
        self.event_handlers = EventHandlers(self)
        self.board_manager = BoardManager()
        # Tab bars that should support wheel-based horizontal scrolling across tabs
        self._wheel_scroll_tab_bars = set()
        
        # Set up the UI.
        self._init_ui()
        
        # Create initial board display (board must exist before tab state restore ever runs)
        self._create_board()
        self._update_all_buttons()

        # Always show the board (we no longer use a board "info" panel).
        self._show_board()
        # Start in preview mode (greyed-out board) until a game is started.
        setattr(self.game_manager, "preview_mode", True)
        self._update_all_buttons()

        # Now that the board exists, sync the current tab once safely
        if hasattr(self, 'board_tabs'):
            self.event_handlers.handle_tab_change(self.board_tabs.currentIndex())

        # Enable keyboard focus
        self.setFocusPolicy(Qt.StrongFocus)
    
        # Global shortcuts for navigating loaded/bot states even when focus is in QLineEdit
        self._shortcut_left = QShortcut(QKeySequence(Qt.Key_Left), self)
        self._shortcut_left.setContext(Qt.ApplicationShortcut)
        self._shortcut_left.activated.connect(self._on_nav_left)

        self._shortcut_right = QShortcut(QKeySequence(Qt.Key_Right), self)
        self._shortcut_right.setContext(Qt.ApplicationShortcut)
        self._shortcut_right.activated.connect(self._on_nav_right)

        # In the Load Game States tab, Up/Down should select rows (like clicking).
        self._shortcut_up = QShortcut(QKeySequence(Qt.Key_Up), self)
        self._shortcut_up.setContext(Qt.ApplicationShortcut)
        self._shortcut_up.activated.connect(lambda: self._move_state_table_selection(-1))

        self._shortcut_down = QShortcut(QKeySequence(Qt.Key_Down), self)
        self._shortcut_down.setContext(Qt.ApplicationShortcut)
        self._shortcut_down.activated.connect(lambda: self._move_state_table_selection(1))

    def _on_nav_left(self):
        """Navigate backward through loaded/bot states."""
        if self.game_manager.current_tab_index == 0:
            self.game_manager.navigate_loaded_state(-1)
        elif self.game_manager.current_tab_index == 2:
            # Route based on the selected bot sub-tab (Logic vs NN tabs).
            try:
                sub = int(self.bots_tabs.currentIndex()) if hasattr(self, "bots_tabs") else 0
            except Exception:
                sub = 0
            if sub == 0:
                self.game_manager.navigate_bot_state(-1)
            elif sub == 1:
                self.game_manager.navigate_nn_mine_state(-1)
            elif sub == 2:
                self.game_manager.navigate_nn_move_state(-1)
            elif sub == 3:
                self.game_manager.navigate_nn_think_state(-1)

    def _on_nav_right(self):
        """Navigate forward through loaded/bot states."""
        if self.game_manager.current_tab_index == 0:
            self.game_manager.navigate_loaded_state(1)
        elif self.game_manager.current_tab_index == 2:
            # Route based on the selected bot sub-tab (Logic vs NN tabs).
            try:
                sub = int(self.bots_tabs.currentIndex()) if hasattr(self, "bots_tabs") else 0
            except Exception:
                sub = 0
            if sub == 0:
                self.game_manager.navigate_bot_state(1)
            elif sub == 1:
                self.game_manager.navigate_nn_mine_state(1)
            elif sub == 2:
                self.game_manager.navigate_nn_move_state(1)
            elif sub == 3:
                self.game_manager.navigate_nn_think_state(1)

    def _move_state_table_selection(self, delta: int):
        """
        Move selection in the Load Game States table up/down by delta rows.
        This is intentionally application-wide so the user can use arrow keys
        without the table needing focus.
        """
        if getattr(self.game_manager, "current_tab_index", None) != 0:
            return
        if not hasattr(self, "state_table") or self.state_table is None:
            return

        row_count = self.state_table.rowCount()
        if row_count <= 0:
            return

        current = self.state_table.currentRow()
        if current < 0:
            current = 0

        new_row = max(0, min(row_count - 1, current + int(delta)))
        self.state_table.setCurrentCell(new_row, 0)
        self.state_table.selectRow(new_row)
        item = self.state_table.item(new_row, 0)
        if item is not None:
            self.state_table.scrollToItem(item, QAbstractItemView.PositionAtCenter)
    
    def _init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Minesweeper")
        self.setStyleSheet(f"background-color: {self.COLORS['background']};")
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setOpaqueResize(True)
        splitter.setHandleWidth(8)
        # Keep a reference so we can react to sidebar dragging (auto-fit cell sizing).
        self.main_splitter = splitter
        splitter.splitterMoved.connect(lambda *_: self._schedule_autofit_cell_size())
        
        # Left panel: Board and tabs
        board_container = self._create_board_panel()
        stats_widget = self._create_stats_panel()
        
        # Set size constraints
        board_container.setMinimumWidth(350)
        splitter.addWidget(board_container)
        splitter.addWidget(stats_widget)
        splitter.setSizes([600, 300])
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 0)

        main_layout.addWidget(splitter)

        # Window setup
        self.setMinimumSize(800, 600)
        self.resize(1000, 700)

    def _create_board_panel(self):
        """Create the board and tabs panel."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)

        # Board stack widget (kept for compatibility, but we no longer show an "info" panel)
        self.board_stack = QStackedWidget()
        
        # Board scroll area
        self.board_widget = QWidget()
        self.board_scroll = QScrollArea()
        self.board_scroll.setWidget(self.board_widget)
        # Keep the board "packed" (button-grid defines size; scrollbars appear as needed)
        self.board_scroll.setWidgetResizable(False)
        self.board_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.board_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.board_scroll.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.board_scroll.setMinimumSize(200, 200)  # Ensure minimum size
        self.board_scroll.setFrameShape(QFrame.NoFrame)
        self.board_scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {self.COLORS['background']};
                border: 2px solid {self.COLORS['border']};
                border-radius: 5px;
            }}
        """)
        
        self.board_stack.addWidget(self.board_scroll)
        # We'll place the board_stack and tabs into a vertical splitter so the user can
        # Adjust how much vertical space the board vs tabs get.
        
        # Tab widget
        self.board_tabs = QTabWidget()
        self.board_tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 2px solid {self.COLORS['border']};
                border-radius: 5px;
                background-color: {self.COLORS['background']};
            }}
            QTabBar::tab {{
                background-color: {self.COLORS['header_bg']};
                color: {self.COLORS['text']};
                padding: 12px 26px;
                border: none;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                margin-right: 2px;
                min-width: 170px;
                font-size: 13px;
            }}
            QTabBar::tab:selected {{
                background-color: {self.COLORS['background']};
                color: {self.COLORS['text']};
                font-weight: bold;
            }}
            QTabBar::tab:!selected {{
                background-color: {self.COLORS['header_bg']};
                color: {self.COLORS['text_secondary']};
            }}
            QTabBar::tab:disabled {{
                background-color: #2A2A2A;
                color: #666666;
            }}
            QTabBar::tab:hover {{
                background-color: #3A3A3A;
            }}
        """)
        
        # Allow horizontal scrolling across tab titles when there are many tabs.
        try:
            tb = self.board_tabs.tabBar()
            tb.setUsesScrollButtons(True)
            tb.setExpanding(False)
            tb.setElideMode(Qt.ElideRight)
            tb.installEventFilter(self)
            self._wheel_scroll_tab_bars.add(tb)
        except Exception:
            pass
        
        # Saved Game States tab
        self.load_state_tab = self._create_load_state_tab()
        self.board_tabs.addTab(self.load_state_tab, "Saved Game States")

        # Playable Game tab
        self.playable_tab = self._create_playable_tab()
        self.board_tabs.addTab(self.playable_tab, "Playable Game")

        # Bots container tab: keeps Saved Game States + Playable Game always visible,
        # While bot sub-tabs (Logic/NN) can scroll independently.
        self.bots_tab = self._create_bots_container_tab()
        self.board_tabs.addTab(self.bots_tab, "Bots")
        self.board_tabs.setTabEnabled(2, False)

        # Default tab
        self.board_tabs.setCurrentIndex(1)

        # Connect tab change handler (DO NOT manually call it here before board exists)
        self.board_tabs.currentChanged.connect(self.event_handlers.handle_tab_change)
        self.game_manager.current_tab_index = 1  # Default to Playable Game

        # Splitter between board and tabs (user-adjustable)
        self.board_tabs_splitter = QSplitter(Qt.Vertical)
        self.board_tabs_splitter.setChildrenCollapsible(False)
        self.board_tabs_splitter.setOpaqueResize(True)
        self.board_tabs_splitter.setHandleWidth(8)
        self.board_tabs_splitter.addWidget(self.board_stack)
        self.board_tabs_splitter.addWidget(self.board_tabs)
        self.board_tabs_splitter.setStretchFactor(0, 3)  # Board gets more space
        self.board_tabs_splitter.setStretchFactor(1, 1)  # Tabs still visible
        # Reasonable defaults
        self.board_stack.setMinimumHeight(280)
        self.board_tabs.setMinimumHeight(240)
        self.board_tabs.setMaximumHeight(520)
        # Make the tabs panel taller by default so most content fits (without adding padding inside tabs).
        self.board_tabs_splitter.setSizes([450, 370])
        # When the user drags the handle, auto-fit cell sizes so the board uses the space well.
        self.board_tabs_splitter.splitterMoved.connect(lambda *_: self._schedule_autofit_cell_size())

        layout.addWidget(self.board_tabs_splitter, 1)

        return container

    def _create_load_state_tab(self):
        """Create the Saved Game States tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        # Keep this tab tight; the table should be the focus.
        layout.setSpacing(6)

        title = QLabel("Saved Game States")
        title.setStyleSheet(f"color: {self.COLORS['text']}; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        
        instructions = QLabel("Select a saved game-states file to load:")
        layout.addWidget(self._make_scrollable_description(instructions, max_height=36))
        
        # State table
        self.state_table = QTableWidget()
        self.state_table.setColumnCount(7)
        self.state_table.setHorizontalHeaderLabels([
            'Last Updated', 'Mode', 'Status', 'Mines Triggered', 'Size', 'States', 'File'
        ])
        self.state_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.state_table.setSelectionMode(QAbstractItemView.SingleSelection)
        # Use explicit column widths + stretch only the filename column.
        header = self.state_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QHeaderView.Interactive)
        # Sorting UX (small arrow toggle next to header text).
        # Handle which columns are actually sortable in EventHandlers.
        try:
            header.setSortIndicatorShown(True)
            header.setSectionsClickable(True)
            header.sectionClicked.connect(self.event_handlers.handle_state_table_header_clicked)
        except Exception:
            pass
        # Force a wide-enough section for the header label (Qt may otherwise size to cell contents).
        header.setSectionResizeMode(3, QHeaderView.Fixed)  # Mines Triggered
        header.setSectionResizeMode(6, QHeaderView.Stretch)  # File
        self.state_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.state_table.clicked.connect(self.event_handlers.handle_state_table_click)
        # Keyboard navigation should behave like clicking: selecting a row loads it.
        self.state_table.itemSelectionChanged.connect(
            self.event_handlers.handle_state_table_selection_changed
        )
        self.state_table.setAlternatingRowColors(True)
        self.state_table.setShowGrid(False)
        # Readability/UX improvements:
        # - Make rows a bit taller (prevents clipped row numbers/text on macOS)
        # - Force scrollbars to take their own space (no overlay)
        self.state_table.setWordWrap(False)
        self.state_table.verticalHeader().setDefaultSectionSize(34)
        self.state_table.setViewportMargins(0, 0, 0, 0)
        self.state_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.state_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.state_table.setStyleSheet(self._get_table_stylesheet())
        # Preserve per-cell foreground colors (e.g., Mode/Status colors) when the row is selected.
        try:
            self.state_table.setItemDelegate(PreserveForegroundOnSelectDelegate(self.state_table))
        except Exception:
            pass
        # Ensure headers remain readable (esp. "Mines Triggered")
        try:
            self.state_table.setColumnWidth(0, 170)  # Last Updated (doesn't need to be huge)
            self.state_table.setColumnWidth(1, 150)  # Mode (fits "NN Task 1/2/3")
            self.state_table.setColumnWidth(2, 80)   # Status
            self.state_table.setColumnWidth(3, 190)  # Mines Triggered (header must fit)
            self.state_table.setColumnWidth(4, 90)   # Size
            self.state_table.setColumnWidth(5, 70)   # States
        except Exception:
            pass
        layout.addWidget(self.state_table)
        
        # Navigation controls
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(10)
        
        nav_label = QLabel("Jump to action:")
        nav_label.setStyleSheet(f"color: {self.COLORS['text']}; font-size: 12px;")
        nav_layout.addWidget(nav_label)
        
        self.state_jump_input = QLineEdit()
        self.state_jump_input.setPlaceholderText("Enter action number (1-based)")
        self.state_jump_input.setMaximumWidth(150)
        self.state_jump_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {self.COLORS['input_bg']};
                color: {self.COLORS['text']};
                border: 1px solid {self.COLORS['border']};
                border-radius: 3px;
                padding: 5px;
                font-size: 12px;
            }}
        """)
        self.state_jump_input.returnPressed.connect(self.event_handlers.handle_jump_input)
        nav_layout.addWidget(self.state_jump_input)

        # Continue playing from a loaded snapshot (enabled only when valid)
        self.continue_play_button = StyledButton("Continue Playing", "secondary")
        self.continue_play_button.setEnabled(False)
        self.continue_play_button.setFixedHeight(36)
        self.continue_play_button.clicked.connect(self.event_handlers.handle_continue_playing_from_loaded)
        nav_layout.addWidget(self.continue_play_button)

        # Delete the last action in the loaded file (enabled only when at last action)
        self.delete_last_action_button = StyledButton("Delete Last Action", "danger")
        self.delete_last_action_button.setEnabled(False)
        self.delete_last_action_button.setFixedHeight(36)
        self.delete_last_action_button.clicked.connect(self.event_handlers.handle_delete_last_action)
        nav_layout.addWidget(self.delete_last_action_button)

        # Delete the entire loaded file (enabled when a file is loaded)
        self.delete_file_button = StyledButton("Delete File", "danger")
        self.delete_file_button.setEnabled(False)
        self.delete_file_button.setFixedHeight(36)
        self.delete_file_button.clicked.connect(self.event_handlers.handle_delete_loaded_file)
        nav_layout.addWidget(self.delete_file_button)
        
        nav_layout.addStretch()
        
        self.state_nav_label = QLabel("No state loaded")
        self.state_nav_label.setStyleSheet(f"color: {self.COLORS['text_secondary']}; font-size: 12px;")
        nav_layout.addWidget(self.state_nav_label)
        
        layout.addLayout(nav_layout)
        layout.addStretch()
        
        # Populate table
        self.event_handlers._populate_state_table()
        
        return tab

    def _create_playable_tab(self):
        """Create the Playable Game tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        # Keep spacing tight; the vertical splitter default height provides room.
        layout.setSpacing(10)
        
        title = QLabel("Playable Game")
        title.setStyleSheet(f"color: {self.COLORS['text']}; font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        info = QLabel(
            "Game generation (Level 2): the board is NOT generated until your first click. "
            "On the first click, the game generates a mine layout that guarantees a safe 3×3 area "
            "(the clicked cell and its 8 neighbors contain no mines), then fills in the clue numbers.\n\n"
            "State files: the JSON file is created only AFTER the first click (first captured state), "
            "because only then does `actual_board_initial` exist.\n\n"
            "How to play: left-click reveals a cell; numbers show how many adjacent mines exist; "
            "right-click places/removes a flag on an unrevealed cell."
        )
        # Fixed-height description to avoid large blank space above Mode controls.
        layout.addWidget(self._make_scrollable_description(info, max_height=110))
        
        # Presets row (not a dropdown) - restored to original Playable tab layout
        presets_row = QHBoxLayout()
        presets_row.setSpacing(10)

        presets_label = QLabel("Mode:")
        presets_label.setStyleSheet(f"color: {self.COLORS['text_secondary']}; font-size: 12px;")
        presets_row.addWidget(presets_label)
        
        self.preset_button_group = QButtonGroup(self)
        self.preset_button_group.setExclusive(True)

        self.preset_buttons = {}
        for name in ["Easy", "Medium", "Hard", "Custom"]:
            btn = StyledButton(name, "secondary")
            btn.setCheckable(True)
            btn.setFixedHeight(34)
            btn.setEnabled(True)
            self.preset_button_group.addButton(btn)
            self.preset_buttons[name] = btn
            presets_row.addWidget(btn)

        presets_row.addStretch()
        layout.addLayout(presets_row)

        # Dimension inputs row
        inputs_row = QHBoxLayout()
        inputs_row.setSpacing(10)

        def _label(txt: str) -> QLabel:
            lab = QLabel(txt)
            lab.setStyleSheet(f"color: {self.COLORS['text_secondary']}; font-size: 12px;")
            return lab

        self.custom_height_input = QLineEdit()
        self.custom_width_input = QLineEdit()
        self.custom_mines_input = QLineEdit()

        for inp in [self.custom_height_input, self.custom_width_input, self.custom_mines_input]:
            inp.setMaximumWidth(90)
            inp.setStyleSheet(f"""
                QLineEdit {{
                    background-color: {self.COLORS['input_bg']};
                    color: {self.COLORS['text']};
                    border: 1px solid {self.COLORS['border']};
                    border-radius: 3px;
                    padding: 5px;
                    font-size: 12px;
                }}
                QLineEdit:disabled {{
                    background-color: #2A2A2A;
                    color: #8A8A8A;
                    border: 1px solid #333333;
                }}
            """)

        # Reasonable thresholds for Custom
        self.custom_height_input.setValidator(QIntValidator(5, 50, self))
        self.custom_width_input.setValidator(QIntValidator(5, 50, self))
        self.custom_mines_input.setValidator(QIntValidator(1, 9999, self))

        inputs_row.addWidget(_label("Height"))
        inputs_row.addWidget(self.custom_height_input)
        inputs_row.addWidget(_label("Width"))
        inputs_row.addWidget(self.custom_width_input)
        inputs_row.addWidget(_label("Mines"))
        inputs_row.addWidget(self.custom_mines_input)

        self.apply_custom_button = StyledButton("Apply", "primary")
        self.apply_custom_button.setFixedHeight(34)
        self.apply_custom_button.setEnabled(False)
        inputs_row.addWidget(self.apply_custom_button)

        inputs_row.addStretch()
        layout.addLayout(inputs_row)
        
        # Continue-after-mine toggle lives in the right-side stats panel (not here).

        # Wire up preset changes
        for name, btn in self.preset_buttons.items():
            btn.clicked.connect(lambda _, n=name: self._select_preset(n))
        
        # Apply custom on button or Enter
        self.apply_custom_button.clicked.connect(self._apply_custom_settings_from_inputs)
        self.custom_height_input.returnPressed.connect(self._apply_custom_settings_from_inputs)
        self.custom_width_input.returnPressed.connect(self._apply_custom_settings_from_inputs)
        self.custom_mines_input.returnPressed.connect(self._apply_custom_settings_from_inputs)

        # Select initial preset based on current settings
        initial = self._infer_preset_name()
        if initial not in self.preset_buttons:
            initial = "Custom"
        self._select_preset(initial)

        layout.addStretch()
        return tab

    def _select_preset(self, preset_name: str):
        """Select a preset and (for non-custom) apply it immediately."""
        from .constants import PRESETS

        self.preset_name = preset_name
        if preset_name in self.preset_buttons:
            self.preset_buttons[preset_name].setChecked(True)

        is_custom = (preset_name == "Custom")

        # Fill values from preset or current
        if not is_custom and preset_name in PRESETS:
            p = PRESETS[preset_name]
            h, w, m = int(p["height"]), int(p["width"]), int(p["num_mines"])
        else:
            # For Custom, restore last-applied custom values if available.
            saved = getattr(self, "_custom_saved_preset", None)
            if is_custom and isinstance(saved, dict):
                h = int(saved.get("height", self.game_manager.height))
                w = int(saved.get("width", self.game_manager.width))
                m = int(saved.get("num_mines", self.game_manager.num_mines))
            else:
                h, w, m = int(self.game_manager.height), int(self.game_manager.width), int(self.game_manager.num_mines)

        self.custom_height_input.setText(str(h))
        self.custom_width_input.setText(str(w))
        self.custom_mines_input.setText(str(m))

        # Lock/unlock inputs
        self.custom_height_input.setEnabled(is_custom)
        self.custom_width_input.setEnabled(is_custom)
        self.custom_mines_input.setEnabled(is_custom)
        self.apply_custom_button.setEnabled(is_custom)

        # Apply immediately for Easy/Medium/Hard; for Custom, user must press Apply.
        if not is_custom and preset_name in PRESETS:
            self.game_manager.apply_settings({"height": h, "width": w, "num_mines": m})
            # Show preview (grey) board until New Game starts
            setattr(self.game_manager, "preview_mode", True)
            self._show_board()
            self._update_all_buttons()
        elif is_custom:
            # When clicking Custom again, auto-apply the remembered custom preset so the board preview
            # Updates just like the other preset buttons.
            if isinstance(getattr(self, "_custom_saved_preset", None), dict):
                self.game_manager.apply_settings({"height": h, "width": w, "num_mines": m})
                setattr(self.game_manager, "preview_mode", True)
                self._show_board()
                self._update_all_buttons()

    def _on_continue_after_mine_toggled(self, _state: int):
        """Toggle whether the current game continues after triggering a mine."""
        enabled = bool(getattr(self, "continue_after_mine_checkbox", None) and self.continue_after_mine_checkbox.isChecked())
        self.game_manager.allow_mine_triggers = enabled
        try:
            setattr(self.game_manager.game, "allow_mine_triggers", enabled)
        except Exception:
            pass
        # Also update the GUI's current game reference (extra safety).
        try:
            setattr(self.game, "allow_mine_triggers", enabled)
        except Exception:
            pass
        if hasattr(self, "state_details"):
            self.state_details.setText(
                "Continue after mine: ON\n\n"
                "Notes:\n"
                "- Triggering a mine will not stop the run.\n"
                "- Mines may appear on the board.\n"
                "- 'Mines Triggered' counts how many mines were hit.\n"
                if enabled
                else
                "Continue after mine: OFF\n\n"
                "Classic Minesweeper rules:\n"
                "- The first mine ends the run."
            )

    def _apply_custom_settings_from_inputs(self):
        """Validate and apply Custom settings."""
        if self.preset_name != "Custom":
            return

        try:
            h = int(self.custom_height_input.text() or "0")
            w = int(self.custom_width_input.text() or "0")
            m = int(self.custom_mines_input.text() or "0")
        except Exception:
            return

        # Basic validation (mines depends on board area)
        if h < 5 or w < 5 or h > 50 or w > 50:
            return
        if m < 1 or m >= h * w:
            return

        self.game_manager.apply_settings({"height": h, "width": w, "num_mines": m})
        # Remember this as the last Custom preset so switching away/back restores it.
        self._custom_saved_preset = {"height": h, "width": w, "num_mines": m}
        setattr(self.game_manager, "preview_mode", True)
        self._show_board()
        self._update_all_buttons()

    def _create_bot_tab(self):
        """Create the Logic Bot tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        title = QLabel("Logic Bot Demo")
        title.setStyleSheet(f"color: {self.COLORS['text']}; font-size: 14px; font-weight: bold;")
        layout.addWidget(title)
        
        self.bot_tab_info = QLabel(
            "Logic bot replay runs on the current board.\n\n"
            "Controls:\n"
            "- Run generates a replay\n"
            "- Use \u2190 \u2192 to step through actions\n"
            "- Use 'Jump to action' to go to a specific step"
        )
        layout.addWidget(self._make_scrollable_description(self.bot_tab_info, max_height=44))

        # Bot selector (future-proof: more logic bots later)
        sel_row = QHBoxLayout()
        sel_row.setSpacing(10)
        sel_lbl = QLabel("Bot:")
        sel_lbl.setStyleSheet(f"color: {self.COLORS['text_secondary']}; font-size: 12px;")
        sel_row.addWidget(sel_lbl)
        self.logic_bot_selector = QComboBox()
        self.logic_bot_selector.setMinimumWidth(260)
        self.logic_bot_selector.setStyleSheet(f"color: {self.COLORS['text']}; background-color: {self.COLORS['input_bg']}; padding: 4px;")
        sel_row.addWidget(self.logic_bot_selector)
        # If the selection changes, refresh enable/disable states.
        try:
            self.logic_bot_selector.currentIndexChanged.connect(lambda _=None: self.event_handlers.handle_bot_subtab_change(0))
        except Exception:
            pass
        sel_row.addStretch()
        layout.addLayout(sel_row)

        self.run_logic_bot_btn = StyledButton("Run Logic Bot", "primary")
        self.run_logic_bot_btn.setEnabled(False)
        self.run_logic_bot_btn.clicked.connect(self.event_handlers.handle_run_logic_bot)
        layout.addWidget(self.run_logic_bot_btn)

        # Bot navigation controls (mirrors Load Game State "Jump to action" UX)
        bot_nav_layout = QHBoxLayout()
        bot_nav_layout.setSpacing(10)

        bot_nav_label = QLabel("Jump to action:")
        bot_nav_label.setStyleSheet(f"color: {self.COLORS['text']}; font-size: 12px;")
        bot_nav_layout.addWidget(bot_nav_label)

        self.bot_jump_input = QLineEdit()
        self.bot_jump_input.setPlaceholderText("Enter action number (1-based)")
        self.bot_jump_input.setMaximumWidth(150)
        self.bot_jump_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {self.COLORS['input_bg']};
                color: {self.COLORS['text']};
                border: 1px solid {self.COLORS['border']};
                border-radius: 3px;
                padding: 6px;
                font-size: 12px;
            }}
        """)
        self.bot_jump_input.returnPressed.connect(self.event_handlers.handle_bot_jump_input)
        bot_nav_layout.addWidget(self.bot_jump_input)

        bot_nav_layout.addStretch()

        self.bot_state_nav_label = QLabel("No bot state loaded")
        self.bot_state_nav_label.setStyleSheet(f"color: {self.COLORS['text_secondary']}; font-size: 12px;")
        bot_nav_layout.addWidget(self.bot_state_nav_label)

        layout.addLayout(bot_nav_layout)

        bot_note = QLabel(
            "Run Logic Bot will generate a game-states JSON run for the currently loaded game board.\n\n"
            "How it works:\n"
            "1) It starts from the same first click as your game (and uses the same underlying mine layout).\n"
            "2) It applies deterministic inference rules from revealed clue numbers to infer safe cells/mines.\n"
            "3) If no safe inference is available, it falls back to choosing a random remaining cell.\n\n"
            "Because of that random fallback, the Logic Bot is NOT fully deterministic—re-running may produce a different path."
        )
        layout.addWidget(self._make_scrollable_description(bot_note, max_height=140))

        layout.addStretch()
        return tab

    def _create_bots_container_tab(self):
        """
        Create the outer "Bots" tab, which contains a nested QTabWidget for individual bot tabs.
        The nested tab bar can scroll; the main tabs remain always visible.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.bots_tabs = QTabWidget()
        # Reuse the same tab styling as the main board tabs.
        self.bots_tabs.setStyleSheet(self.board_tabs.styleSheet())

        try:
            tb = self.bots_tabs.tabBar()
            tb.setUsesScrollButtons(True)
            tb.setExpanding(False)
            tb.setElideMode(Qt.ElideRight)
            tb.installEventFilter(self)  # Enable wheel scrolling across bot tabs too
            self._wheel_scroll_tab_bars.add(tb)
        except Exception:
            pass

        # Logic Bot tab
        self.bot_tab = self._create_bot_tab()
        self.bots_tabs.addTab(self.bot_tab, "Logic Bot")

        # Neural Network bot tabs (mirror Logic Bot UX)
        self.nn_mine_tab = self._create_nn_mine_tab()
        self.bots_tabs.addTab(self.nn_mine_tab, "NN: Mine Prediction")

        self.nn_move_tab = self._create_nn_move_tab()
        self.bots_tabs.addTab(self.nn_move_tab, "NN: Move Prediction")

        self.nn_think_tab = self._create_nn_think_tab()
        self.bots_tabs.addTab(self.nn_think_tab, "NN: Thinking Deeper")

        # When switching between bot sub-tabs, delegate to EventHandlers
        try:
            self.bots_tabs.currentChanged.connect(self.event_handlers.handle_bot_subtab_change)
        except Exception:
            pass

        layout.addWidget(self.bots_tabs)
        return tab

    def _create_nn_mine_tab(self):
        """Create the NN Mine Prediction tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        title = QLabel("Neural Net Bot — Mine Prediction")
        title.setStyleSheet(f"color: {self.COLORS['text']}; font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        info = QLabel(
            "Task 1 (Mine Prediction)\n\n"
            "What I'm doing:\n"
            "- I predict mine probabilities for all unrevealed cells\n"
            "- Then I click the cell I think is safest\n\n"
            "Controls:\n"
            "- Use \u2190 \u2192 to step through actions\n"
            "- Use 'Jump to action' to go to a specific step"
        )
        layout.addWidget(self._make_scrollable_description(info, max_height=64))

        sel_row = QHBoxLayout()
        sel_row.setSpacing(10)
        sel_lbl = QLabel("Model:")
        sel_lbl.setStyleSheet(f"color: {self.COLORS['text_secondary']}; font-size: 12px;")
        sel_row.addWidget(sel_lbl)
        self.nn_mine_model_selector = QComboBox()
        self.nn_mine_model_selector.setMinimumWidth(260)
        self.nn_mine_model_selector.setStyleSheet(f"color: {self.COLORS['text']}; background-color: {self.COLORS['input_bg']}; padding: 4px;")
        sel_row.addWidget(self.nn_mine_model_selector)
        try:
            self.nn_mine_model_selector.currentIndexChanged.connect(lambda _=None: self.event_handlers.handle_bot_subtab_change(1))
        except Exception:
            pass
        sel_row.addStretch()
        layout.addLayout(sel_row)

        self.nn_mine_requirements_label = QLabel("")
        layout.addWidget(self._make_scrollable_description(self.nn_mine_requirements_label, max_height=44))

        self.run_nn_mine_btn = StyledButton("Run NN Bot (Mine Prediction)", "primary")
        self.run_nn_mine_btn.setEnabled(False)
        self.run_nn_mine_btn.clicked.connect(self.event_handlers.handle_run_nn_mine_bot)
        layout.addWidget(self.run_nn_mine_btn)

        nav = QHBoxLayout()
        nav.setSpacing(10)
        lbl = QLabel("Jump to action:")
        lbl.setStyleSheet(f"color: {self.COLORS['text']}; font-size: 12px;")
        nav.addWidget(lbl)

        self.nn_mine_jump_input = QLineEdit()
        self.nn_mine_jump_input.setPlaceholderText("Enter action number (1-based)")
        self.nn_mine_jump_input.setMaximumWidth(150)
        self.nn_mine_jump_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {self.COLORS['input_bg']};
                color: {self.COLORS['text']};
                border: 1px solid {self.COLORS['border']};
                border-radius: 3px;
                padding: 6px;
                font-size: 12px;
            }}
        """)
        self.nn_mine_jump_input.returnPressed.connect(self.event_handlers.handle_nn_mine_jump_input)
        nav.addWidget(self.nn_mine_jump_input)
        nav.addStretch()

        self.nn_mine_state_nav_label = QLabel("No NN state loaded")
        self.nn_mine_state_nav_label.setStyleSheet(f"color: {self.COLORS['text_secondary']}; font-size: 12px;")
        nav.addWidget(self.nn_mine_state_nav_label)

        layout.addLayout(nav)
        layout.addStretch()
        return tab

    def _create_nn_move_tab(self):
        """Create the NN Move Prediction tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        title = QLabel("Neural Net Bot — Move Prediction (Actor/Critic)")
        title.setStyleSheet(f"color: {self.COLORS['text']}; font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        info = QLabel(
            "Task 2 (Actor/Critic move selection)\n\n"
            "What I'm doing:\n"
            "- I score candidate clicks with my Task 2 model (predicted survivability)\n"
            "- Then I take the highest-scoring click (subject to the preset/model)\n\n"
            "Controls:\n"
            "- Use \u2190 \u2192 to step through actions\n"
            "- Use 'Jump to action' to go to a specific step"
        )
        layout.addWidget(self._make_scrollable_description(info, max_height=64))

        sel_row = QHBoxLayout()
        sel_row.setSpacing(10)
        sel_lbl = QLabel("Model:")
        sel_lbl.setStyleSheet(f"color: {self.COLORS['text_secondary']}; font-size: 12px;")
        sel_row.addWidget(sel_lbl)
        self.nn_move_model_selector = QComboBox()
        self.nn_move_model_selector.setMinimumWidth(260)
        self.nn_move_model_selector.setStyleSheet(f"color: {self.COLORS['text']}; background-color: {self.COLORS['input_bg']}; padding: 4px;")
        sel_row.addWidget(self.nn_move_model_selector)
        try:
            self.nn_move_model_selector.currentIndexChanged.connect(lambda _=None: self.event_handlers.handle_bot_subtab_change(2))
        except Exception:
            pass
        sel_row.addStretch()
        layout.addLayout(sel_row)

        self.nn_move_requirements_label = QLabel("")
        # Keep this taller because I also display per-model stats here (not just preset requirements).
        layout.addWidget(self._make_scrollable_description(self.nn_move_requirements_label, max_height=120))

        self.run_nn_move_btn = StyledButton("Run NN Bot (Move Prediction)", "primary")
        self.run_nn_move_btn.setEnabled(False)
        self.run_nn_move_btn.clicked.connect(self.event_handlers.handle_run_nn_move_bot)
        layout.addWidget(self.run_nn_move_btn)

        nav = QHBoxLayout()
        nav.setSpacing(10)
        lbl = QLabel("Jump to action:")
        lbl.setStyleSheet(f"color: {self.COLORS['text']}; font-size: 12px;")
        nav.addWidget(lbl)

        self.nn_move_jump_input = QLineEdit()
        self.nn_move_jump_input.setPlaceholderText("Enter action number (1-based)")
        self.nn_move_jump_input.setMaximumWidth(150)
        self.nn_move_jump_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {self.COLORS['input_bg']};
                color: {self.COLORS['text']};
                border: 1px solid {self.COLORS['border']};
                border-radius: 3px;
                padding: 6px;
                font-size: 12px;
            }}
        """)
        self.nn_move_jump_input.returnPressed.connect(self.event_handlers.handle_nn_move_jump_input)
        nav.addWidget(self.nn_move_jump_input)
        nav.addStretch()

        self.nn_move_state_nav_label = QLabel("No NN state loaded")
        self.nn_move_state_nav_label.setStyleSheet(f"color: {self.COLORS['text_secondary']}; font-size: 12px;")
        nav.addWidget(self.nn_move_state_nav_label)

        layout.addLayout(nav)
        layout.addStretch()
        return tab

    def _create_nn_think_tab(self):
        """Create the NN Thinking Deeper tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        title = QLabel("Neural Net Bot — Thinking Deeper")
        title.setStyleSheet(f"color: {self.COLORS['text']}; font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        info = QLabel(
            "Task 3 (Thinking Deeper)\n\n"
            "What I'm doing:\n"
            "- I run the same reasoning block multiple times (thinking steps)\n"
            "- Then I click the cell with the lowest predicted mine probability\n\n"
            "Controls:\n"
            "- Use \u2190 \u2192 to step through actions\n"
            "- Use 'Jump to action' to go to a specific step"
        )
        layout.addWidget(self._make_scrollable_description(info, max_height=64))

        sel_row = QHBoxLayout()
        sel_row.setSpacing(10)
        sel_lbl = QLabel("Model:")
        sel_lbl.setStyleSheet(f"color: {self.COLORS['text_secondary']}; font-size: 12px;")
        sel_row.addWidget(sel_lbl)
        self.nn_think_model_selector = QComboBox()
        self.nn_think_model_selector.setMinimumWidth(260)
        self.nn_think_model_selector.setStyleSheet(f"color: {self.COLORS['text']}; background-color: {self.COLORS['input_bg']}; padding: 4px;")
        sel_row.addWidget(self.nn_think_model_selector)
        try:
            self.nn_think_model_selector.currentIndexChanged.connect(lambda _=None: self.event_handlers.handle_bot_subtab_change(3))
        except Exception:
            pass
        sel_row.addStretch()
        layout.addLayout(sel_row)

        self.nn_think_requirements_label = QLabel("")
        layout.addWidget(self._make_scrollable_description(self.nn_think_requirements_label, max_height=44))

        self.run_nn_think_btn = StyledButton("Run NN Bot (Thinking Deeper)", "primary")
        self.run_nn_think_btn.setEnabled(False)
        self.run_nn_think_btn.clicked.connect(self.event_handlers.handle_run_nn_think_bot)
        layout.addWidget(self.run_nn_think_btn)

        nav = QHBoxLayout()
        nav.setSpacing(10)
        lbl = QLabel("Jump to action:")
        lbl.setStyleSheet(f"color: {self.COLORS['text']}; font-size: 12px;")
        nav.addWidget(lbl)

        self.nn_think_jump_input = QLineEdit()
        self.nn_think_jump_input.setPlaceholderText("Enter action number (1-based)")
        self.nn_think_jump_input.setMaximumWidth(150)
        self.nn_think_jump_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {self.COLORS['input_bg']};
                color: {self.COLORS['text']};
                border: 1px solid {self.COLORS['border']};
                border-radius: 3px;
                padding: 6px;
                font-size: 12px;
            }}
        """)
        self.nn_think_jump_input.returnPressed.connect(self.event_handlers.handle_nn_think_jump_input)
        nav.addWidget(self.nn_think_jump_input)
        nav.addStretch()

        self.nn_think_state_nav_label = QLabel("No NN state loaded")
        self.nn_think_state_nav_label.setStyleSheet(f"color: {self.COLORS['text_secondary']}; font-size: 12px;")
        nav.addWidget(self.nn_think_state_nav_label)

        layout.addLayout(nav)
        layout.addStretch()
        return tab

    def _make_scrollable_description(self, label: QLabel, *, max_height: int = 140) -> QScrollArea:
        """
        Wrap a QLabel into a small scrollable region.
        This keeps tab layouts stable while allowing long descriptions to scroll.
        """
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        label.setStyleSheet(f"color: {self.COLORS['text_secondary']}; font-size: 11px;")

        sc = QScrollArea()
        sc.setWidget(label)
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.NoFrame)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sc.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # Shrink-to-fit up to max_height; scroll only if needed.
        sc.setMaximumHeight(int(max_height))
        sc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        sc.setStyleSheet("QScrollArea { background: transparent; }")
        return sc
    
    def _create_stats_panel(self):
        """Create the statistics and controls panel."""
        stats_widget = QFrame()
        stats_widget.setStyleSheet(f"""
            QFrame {{
                background-color: {self.COLORS['background']};
                border: 2px solid #424242;
                border-radius: 5px;
            }}
        """)
        stats_layout = QVBoxLayout(stats_widget)
        stats_layout.setSpacing(15)
        stats_layout.setContentsMargins(20, 20, 20, 20)
        
        # Game state label
        self.state_label = QLabel("Game State: Ready")
        self.state_label.setStyleSheet(f"""
            QLabel {{
                color: {self.COLORS['text']};
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                background-color: #2E2E2E;
                border-radius: 5px;
            }}
        """)
        self.state_label.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(self.state_label)
        
        # Wrapped details area under the main status ("dialog") label
        self.state_details = QLabel("")
        self.state_details.setWordWrap(True)
        self.state_details.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.state_details.setStyleSheet(f"""
            QLabel {{
                color: {self.COLORS['text_secondary']};
                font-size: 12px;
                padding: 10px;
                background-color: #1F1F1F;
                border-radius: 5px;
            }}
        """)
        stats_layout.addWidget(self.state_details)

        # Statistics rows
        moves_row = self._create_stat_row("Moves", "0")
        cells_opened_row = self._create_stat_row("Cells Opened", "0")
        mines_triggered_row = self._create_stat_row("Mines Triggered", "0")
        flags_row = self._create_stat_row("Flags Placed", "0")
        mines_remaining_row = self._create_stat_row("Mines Remaining", str(self.game.mine_count))
        cells_remaining_row = self._create_stat_row("Cells Remaining", "0")
        progress_row = self._create_stat_row("Progress", "0%")

        # Add the row widgets to the layout
        stats_layout.addWidget(moves_row)
        stats_layout.addWidget(cells_opened_row)
        stats_layout.addWidget(mines_triggered_row)
        stats_layout.addWidget(flags_row)
        stats_layout.addWidget(mines_remaining_row)
        stats_layout.addWidget(cells_remaining_row)
        stats_layout.addWidget(progress_row)

        # Continue-after-mine toggle (moved here from the Playable tab)
        self.continue_after_mine_checkbox = QCheckBox("Continue after mine?")
        self.continue_after_mine_checkbox.setChecked(False)
        self.continue_after_mine_checkbox.setStyleSheet(f"color: {self.COLORS['text_secondary']}; font-size: 12px;")
        self.continue_after_mine_checkbox.stateChanged.connect(self._on_continue_after_mine_toggled)
        stats_layout.addWidget(self.continue_after_mine_checkbox)

        # Tab availability note (I keep it short; enabled/disabled status is enforced in code)
        self.tab_availability_label = QLabel("")
        self.tab_availability_label.setWordWrap(True)
        self.tab_availability_label.setStyleSheet(
            f"color: {self.COLORS['text_secondary']}; font-size: 11px; padding: 6px;"
        )
        stats_layout.addWidget(self.tab_availability_label)
        
        stats_layout.addStretch()
        
        # Control buttons
        new_game_btn = StyledButton("New Game", 'primary')
        new_game_btn.clicked.connect(self.event_handlers.handle_new_game)
        stats_layout.addWidget(new_game_btn)
        
        # Navigation info
        self.nav_label = QLabel("")
        self.nav_label.setStyleSheet(f"""
            QLabel {{
                color: {self.COLORS['text']};
                font-size: 12px;
                font-weight: bold;
                padding: 10px;
                background-color: #2E2E2E;
                border-radius: 5px;
            }}
        """)
        self.nav_label.setAlignment(Qt.AlignCenter)
        self.nav_label.hide()
        stats_layout.addWidget(self.nav_label)
        
        # Instructions
        instructions = QLabel("Instructions:\n• Left Click: Reveal cell\n• Right Click: Place/Remove flag\n• ← → Arrow keys: Navigate bot/loaded states")
        instructions.setStyleSheet(f"color: {self.COLORS['text_secondary']}; font-size: 11px; padding: 10px;")
        stats_layout.addWidget(instructions)
        
        return stats_widget
    
    def _create_stat_row(self, label_text: str, value_text: str):
        """Create a statistics row."""
        from PyQt5.QtWidgets import QWidget

        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(f"{label_text}:")
        label.setStyleSheet(f"color: {self.COLORS['text_secondary']}; font-size: 12px;")
        row_layout.addWidget(label)

        row_layout.addStretch()

        value_label = QLabel(value_text)
        value_label.setStyleSheet(f"color: {self.COLORS['text']}; font-size: 12px; font-weight: bold;")
        row_layout.addWidget(value_label)

        # Store reference for updates
        attr_name = f"{label_text.lower().replace(' ', '_')}_label"
        setattr(self, attr_name, value_label)

        return row_widget

    def _get_table_stylesheet(self):
        """Get table stylesheet."""
        return f"""
            QTableWidget {{
                background-color: {self.COLORS['background']};
                    color: {self.COLORS['text']};
                border: 2px solid {self.COLORS['border']};
                    border-radius: 5px;
                gridline-color: {self.COLORS['border']};
                font-size: 12px;
                /* Keep selection from overriding per-cell foreground colors (Status colors). */
                selection-background-color: transparent;
            }}
            QTableWidget::item {{
                padding: 8px;
                border: none;
            }}
            QTableWidget::item:alternate {{
                background-color: {self.COLORS['row_alt']};
            }}
            /* Selection: show a thick blue border (no fill) so text colors remain visible. */
            QTableWidget::item:selected,
            QTableWidget::item:selected:active,
            QTableWidget::item:selected:!active,
            QTableWidget::item:selected:alternate {{
                background-color: transparent;
                border-top: 3px solid #2196F3;
                border-bottom: 3px solid #2196F3;
                border-left: 3px solid #2196F3;
                border-right: 3px solid #2196F3;
            }}
            QHeaderView::section {{
                background-color: {self.COLORS['header_bg']};
                    color: {self.COLORS['text']};
                    padding: 10px;
                border: none;
                border-bottom: 2px solid {self.COLORS['border']};
                    font-weight: bold;
                font-size: 13px;
            }}
            QTableCornerButton::section {{
                background-color: {self.COLORS['header_bg']};
                border: none;
                }}
        """

    def eventFilter(self, obj, event):
        """
        Allow horizontal scrolling across overflowing tabs using mouse wheel / trackpad.
        Qt's built-in scroll buttons exist, but wheel events don't always scroll the tab bar.
        """
        try:
            if obj in getattr(self, "_wheel_scroll_tab_bars", set()) and event.type() == QEvent.Wheel:
                bar = obj
                # Prefer horizontal wheel delta; fall back to vertical wheel delta.
                delta = 0
                try:
                    delta = int(event.angleDelta().x())
                except Exception:
                    delta = 0
                if delta == 0:
                    try:
                        delta = int(event.angleDelta().y())
                    except Exception:
                        delta = 0
                if delta != 0:
                    cur = int(bar.currentIndex())
                    count = int(bar.count())
                    # Wheel down/right -> next tab; wheel up/left -> previous tab
                    step = 1 if delta < 0 else -1
                    nxt = max(0, min(count - 1, cur + step))
                    if nxt != cur:
                        bar.setCurrentIndex(nxt)
                    event.accept()
                    return True
        except Exception:
            pass
        return super().eventFilter(obj, event)

    def _infer_preset_name(self):
        """Infer preset name from current settings."""
        current = {
            'height': self.height,
            'width': self.width,
            'num_mines': self.num_mines,
        }

        for name, preset in PRESETS.items():
            if preset == current:
                return name
        return "Custom"

    def _show_board_info(self, message: str = ""):
        """Board info panel has been removed; route any info to the right-side details area."""
        if message and hasattr(self, "state_details"):
            self.state_details.setText(message)
        self._show_board()

    def _show_board(self):
        """Show the board instead of the info panel."""
        if hasattr(self, 'board_stack'):
            self.board_stack.setCurrentIndex(0)
        self._schedule_autofit_cell_size()

    def resizeEvent(self, event):
        """Auto-fit cell sizes on window resize (bounded by CELL_MIN/CELL_MAX)."""
        super().resizeEvent(event)
        self._schedule_autofit_cell_size()

    def _schedule_autofit_cell_size(self):
        """Debounce auto-fit to avoid doing work repeatedly during live resizes."""
        if getattr(self, "_autofit_pending", False):
            return
        self._autofit_pending = True
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, self._autofit_cell_size_now)

    def _autofit_cell_size_now(self):
        """Compute and apply a new cell size based on available board viewport space."""
        self._autofit_pending = False

        # Auto-fit when the board is visible (we always show the board now).
        if not hasattr(self, "board_stack"):
            return
        if not hasattr(self, "board_scroll") or self.board_scroll is None:
            return
        if not hasattr(self, "game_manager") or not getattr(self.game_manager, "cell_buttons", None):
            return

        vp = self.board_scroll.viewport()
        if vp is None:
            return
        vw = max(1, int(vp.width()))
        vh = max(1, int(vp.height()))

        # Fit the grid into the viewport if possible; otherwise clamp at CELL_MIN and allow scroll.
        # IMPORTANT: use board dimensions, not window dimensions.
        cols = int(getattr(self.game_manager, "width", self.width) or self.width)
        rows = int(getattr(self.game_manager, "height", self.height) or self.height)
        cols = max(1, cols)
        rows = max(1, rows)

        target = int(min(vw // cols, vh // rows))
        target = max(self.CELL_MIN, min(self.CELL_MAX, target))

        if target == self.cell_size:
            return

        self.cell_size = target
        self.game_manager.cell_size = target

        # Apply size to existing buttons and refresh styles (font sizes depend on cell_size).
        for btn in self.game_manager.cell_buttons.values():
            try:
                btn.setFixedSize(target, target)
            except Exception:
                pass

        try:
            if self.board_widget is not None and self.board_widget.layout() is not None:
                self.board_widget.layout().invalidate()
            if self.board_widget is not None:
                self.board_widget.adjustSize()
        except Exception:
            pass

        self._update_all_buttons()

    def keyPressEvent(self, event):
        """Handle keyboard events."""
        self.event_handlers.handle_key_press(event)

    def _apply_pending_settings(self):
        """Apply pending settings changes."""
        if hasattr(self, '_pending_settings'):
            settings = self._pending_settings
            del self._pending_settings
            self.game_manager.apply_settings(settings)

    # Delegate board management methods
    def _create_board(self):
        """Create the game board."""
        self.board_manager._create_board(self)

    def _rebuild_board_layout(self):
        """Rebuild the board layout."""
        self.board_manager._rebuild_board_layout(self)

    def _recreate_board_only(self, synchronous=False):
        """Recreate the board UI."""
        self.board_manager._recreate_board_only(self, synchronous)

    def _update_button(self, row: int, col: int):
        """Update a button's appearance."""
        self.board_manager._update_button(self, row, col)

    def _update_all_buttons(self):
        """Update all buttons."""
        self.board_manager._update_all_buttons(self)


def play_game(collect_states: bool = True, state_collector=None):
    """
    Convenience function to start a Minesweeper game.
    
    Args:
        collect_states: If True, collect states for training
        state_collector: Optional StateCollector instance
    """
    import sys
    app = QApplication(sys.argv)
    game = MinesweeperGUI(collect_states, state_collector)
    game.show()
    sys.exit(app.exec_())
