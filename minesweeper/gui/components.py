"""
These are the small reusable UI components I use across the Minesweeper GUI.
"""

from PyQt5.QtWidgets import QPushButton


class StyledButton(QPushButton):
    """This is my reusable styled button component (so I don't repeat stylesheet strings everywhere)."""
    
    STYLES = {
        'primary': """
            QPushButton {
                background-color: #4CAF50;
                color: #000000;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:disabled {
                background-color: #3A3A3A;
                color: #9A9A9A;
            }
            QPushButton:hover {
                background-color: #45A049;
            }
            QPushButton:pressed {
                background-color: #3D8B40;
            }
            QPushButton:checked {
                background-color: #80CBC4;
                color: #000000;
            }
        """,
        'secondary': """
            QPushButton {
                background-color: #2196F3;
                color: #FFFFFF;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:disabled {
                background-color: #3A3A3A;
                color: #9A9A9A;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
            QPushButton:checked {
                background-color: #80CBC4;
                color: #000000;
            }
        """,
        'danger': """
            QPushButton {
                background-color: #F44336;
                color: #FFFFFF;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:disabled {
                background-color: #3A3A3A;
                color: #9A9A9A;
            }
            QPushButton:hover {
                background-color: #D32F2F;
            }
            QPushButton:pressed {
                background-color: #C62828;
            }
            QPushButton:checked {
                background-color: #80CBC4;
                color: #000000;
            }
        """
    }
    
    def __init__(self, text: str, style: str = 'primary', parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(self.STYLES.get(style, self.STYLES['primary']))

