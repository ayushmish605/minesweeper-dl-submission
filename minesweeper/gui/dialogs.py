"""
This file contains the dialog windows I use in the Minesweeper GUI (settings, etc.).
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QSpinBox)
from PyQt5.QtCore import Qt
from .components import StyledButton
from .constants import PRESETS


class GameSettingsDialog(QDialog):
    """I use this dialog to configure game parameters (preset, board size, mine count, cell size)."""
    
    # I match the main GUI color scheme here.
    COLORS = {
        'background': '#1E1E1E',
        'text': '#FFFFFF',
        'text_secondary': '#B0B0B0',
        'border': '#424242',
        'input_bg': '#2E2E2E',
        'input_border': '#555555',
    }
    
    def __init__(self, current_height=22, current_width=22, current_mines=80, 
                 current_cell_size=25, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Game Settings")
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        
        # I apply the dark theme.
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {self.COLORS['background']};
                color: {self.COLORS['text']};
            }}
            QLabel {{
                color: {self.COLORS['text']};
                font-size: 13px;
            }}
            QComboBox {{
                background-color: {self.COLORS['input_bg']};
                color: {self.COLORS['text']};
                border: 2px solid {self.COLORS['input_border']};
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
                min-height: 20px;
            }}
            QComboBox:hover {{
                border-color: #80CBC4;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 30px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid {self.COLORS['text']};
                margin-right: 10px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {self.COLORS['input_bg']};
                color: {self.COLORS['text']};
                selection-background-color: #80CBC4;
                selection-color: #000000;
                border: 2px solid {self.COLORS['input_border']};
            }}
            QSpinBox {{
                background-color: {self.COLORS['input_bg']};
                color: {self.COLORS['text']};
                border: 2px solid {self.COLORS['input_border']};
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
                min-width: 80px;
            }}
            QSpinBox:hover {{
                border-color: #80CBC4;
            }}
            QSpinBox::up-button, QSpinBox::down-button {{
                background-color: {self.COLORS['input_border']};
                border: none;
                width: 20px;
            }}
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
                background-color: #80CBC4;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Title section
        title_label = QLabel("Game Settings")
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {self.COLORS['text']};
                font-size: 20px;
                font-weight: bold;
                padding-bottom: 10px;
            }}
        """)
        layout.addWidget(title_label)
        
        # Difficulty preset selector
        preset_label = QLabel("Difficulty Preset:")
        preset_label.setStyleSheet(f"font-weight: bold; font-size: 14px; color: {self.COLORS['text']};")
        layout.addWidget(preset_label)
        
        self.difficulty_combo = QComboBox()
        self.difficulty_combo.addItems(['Easy', 'Medium', 'Hard', 'Custom'])
        self.difficulty_combo.currentTextChanged.connect(self._on_difficulty_changed)
        layout.addWidget(self.difficulty_combo)
        
        layout.addSpacing(10)
        
        # Custom parameters section
        params_label = QLabel("Custom Parameters:")
        params_label.setStyleSheet(f"font-weight: bold; font-size: 14px; color: {self.COLORS['text']};")
        layout.addWidget(params_label)
        
        params_layout = QVBoxLayout()
        params_layout.setSpacing(15)
        
        # Height
        height_layout = QHBoxLayout()
        height_label = QLabel("Height:")
        height_label.setMinimumWidth(150)
        height_label.setStyleSheet(f"color: {self.COLORS['text_secondary']};")
        height_layout.addWidget(height_label)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(5, 50)
        self.height_spin.setValue(current_height)
        height_layout.addWidget(self.height_spin)
        height_layout.addStretch()
        params_layout.addLayout(height_layout)
        
        # Width
        width_layout = QHBoxLayout()
        width_label = QLabel("Width:")
        width_label.setMinimumWidth(150)
        width_label.setStyleSheet(f"color: {self.COLORS['text_secondary']};")
        width_layout.addWidget(width_label)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(5, 50)
        self.width_spin.setValue(current_width)
        width_layout.addWidget(self.width_spin)
        width_layout.addStretch()
        params_layout.addLayout(width_layout)
        
        # Mines
        mines_layout = QHBoxLayout()
        mines_label = QLabel("Number of Mines:")
        mines_label.setMinimumWidth(150)
        mines_label.setStyleSheet(f"color: {self.COLORS['text_secondary']};")
        mines_layout.addWidget(mines_label)
        self.mines_spin = QSpinBox()
        self.mines_spin.setRange(1, 1000)
        self.mines_spin.setValue(current_mines)
        mines_layout.addWidget(self.mines_spin)
        mines_layout.addStretch()
        params_layout.addLayout(mines_layout)
        
        # Cell size
        cell_size_layout = QHBoxLayout()
        cell_size_label = QLabel("Cell Size (pixels):")
        cell_size_label.setMinimumWidth(150)
        cell_size_label.setStyleSheet(f"color: {self.COLORS['text_secondary']};")
        cell_size_layout.addWidget(cell_size_label)
        self.cell_size_spin = QSpinBox()
        self.cell_size_spin.setRange(15, 50)
        self.cell_size_spin.setValue(current_cell_size)
        cell_size_layout.addWidget(self.cell_size_spin)
        cell_size_layout.addStretch()
        params_layout.addLayout(cell_size_layout)
        
        layout.addLayout(params_layout)
        layout.addStretch()
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        cancel_btn = StyledButton("Cancel", 'danger')
        cancel_btn.clicked.connect(self.reject)
        ok_btn = StyledButton("Apply", 'primary')
        ok_btn.clicked.connect(self._apply_and_close)
        button_layout.addWidget(cancel_btn)
        button_layout.addSpacing(10)
        button_layout.addWidget(ok_btn)
        layout.addLayout(button_layout)
        
        # Guard flag to prevent recursion
        self._updating = False
        
        # Connect spinboxes to update preset selection when manually edited
        self.height_spin.valueChanged.connect(self._update_difficulty_selection)
        self.width_spin.valueChanged.connect(self._update_difficulty_selection)
        self.mines_spin.valueChanged.connect(self._update_difficulty_selection)
        
        # I set initial difficulty based on current values
        self._update_difficulty_selection()
    
    def _update_difficulty_selection(self):
        """Update difficulty combo to match current values (ignores cell_size)."""
        if self._updating:
            return
        
        current = {
            'height': self.height_spin.value(),
            'width': self.width_spin.value(),
            'num_mines': self.mines_spin.value(),
        }
        
        preset_name = "Custom"
        for name, preset in PRESETS.items():
            if preset == current:
                preset_name = name
                break
        
        # Only set if different (prevents recursion)
        if self.difficulty_combo.currentText() != preset_name:
            self._updating = True
            self.difficulty_combo.setCurrentText(preset_name)
            self._updating = False
    
    def _on_difficulty_changed(self, text: str):
        """Update parameters when difficulty preset changes."""
        if self._updating:
            return
        if text == 'Custom':
            return
        
        preset = PRESETS[text]
        
        self._updating = True
        # I update spinbox values and ensure they're committed immediately
        self.height_spin.setValue(preset['height'])
        self.height_spin.interpretText()  # Commit the value
        self.width_spin.setValue(preset['width'])
        self.width_spin.interpretText()  # Commit the value
        self.mines_spin.setValue(preset['num_mines'])
        self.mines_spin.interpretText()  # Commit the value
        # Note: cell_size is not changed by presets, it's a UI preference
        self._updating = False
        
        # I force UI update to show new values immediately
        self.height_spin.update()
        self.width_spin.update()
        self.mines_spin.update()
    
    def _apply_and_close(self):
        """Apply settings and close dialog, ensuring popups are closed and values are committed."""
        # If the dropdown is open, close it so the click isn't "used up" by the popup
        self.difficulty_combo.hidePopup()
        
        # I ensure all spinbox values are committed and up-to-date
        # This is especially important if a preset was just selected
        # The values should already be committed from _on_difficulty_changed,
        # But we ensure they're committed here as well for safety
        self.height_spin.interpretText()
        self.width_spin.interpretText()
        self.mines_spin.interpretText()
        self.cell_size_spin.interpretText()
        
        self.accept()
    
    def get_settings(self):
        """Get the selected settings."""
        return {
            'height': self.height_spin.value(),
            'width': self.width_spin.value(),
            'num_mines': self.mines_spin.value(),
            'cell_size': self.cell_size_spin.value()
        }
