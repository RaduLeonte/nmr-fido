from PySide6.QtCore import QSize
from PySide6.QtWidgets import QPushButton

class ToolbarButton(QPushButton):
    def __init__(self, text: str = "", icon=None, parent=None):
        super().__init__(parent)
        if icon is not None:
            self.setIcon(icon)
            self.setText("")
        else:
            self.setText(text)
        
        self.setIconSize(QSize(24, 24))  # adjust as needed
        
        margin_vertical = 5
        margin_horizontal = 10
        self.setStyleSheet(f"padding: {margin_vertical}px {margin_horizontal}px {margin_vertical}px {margin_horizontal}px;")

    def sizeHint(self) -> QSize:
        size = super().sizeHint()

        min_width = self.fontMetrics().horizontalAdvance(self.text()) + 20  # 20px padding approx

        if not self.text() and not self.icon().isNull():
            min_width = self.iconSize().width() + 4
        
        width = max(min_width, size.width())
        return QSize(width, size.height())

