"""
These are small Qt item delegates I use to keep the GUI rendering consistent.

I keep them centralized because selection/paint behavior can get surprisingly finicky in Qt,
and I don't want that logic scattered across unrelated UI files.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QPalette
from PyQt5.QtWidgets import QStyledItemDelegate, QStyleOptionViewItem


class PreserveForegroundOnSelectDelegate(QStyledItemDelegate):
    """
    I use this delegate to preserve per-item foreground colors when the row/item is selected.

    By default, Qt replaces the text color with the palette's HighlightedText color on selection,
    which makes my "Status" / "Mode" color-coding disappear.
    """

    def paint(self, painter, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)

        fg = index.data(Qt.ForegroundRole)
        if fg is not None:
            # ForegroundRole can be a QBrush or QColor. Normalize to QBrush.
            brush = fg if isinstance(fg, QBrush) else QBrush(fg)
            opt.palette.setBrush(QPalette.Text, brush)
            opt.palette.setBrush(QPalette.HighlightedText, brush)

        super().paint(painter, opt, index)


