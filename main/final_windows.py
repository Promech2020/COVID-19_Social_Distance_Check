import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot, Qt
import cv2

class Final(QWidget):
    app = QApplication(sys.argv)
    def __init__(self, title, message):
        super().__init__()
        self.title = title
        self.left = 10
        self.top = 10
        self.width = 320
        self.height = 200
        self.message = message
        self.finalUI()
        self.setAttribute(Qt.WA_DeleteOnClose)
        
        
    def finalUI(self):
        self.setGeometry(self.left, self.top, self.width, self.height)
        final_msg = QMessageBox()
        final_msg.setWindowTitle(self.title)
        final_msg.setText(self.message)
        if "failed" in self.title.lower():
            final_msg.setIcon(QMessageBox.Critical)
        else:
            final_msg.setIcon(QMessageBox.Information)
        final_msg.setStandardButtons(QMessageBox.Ok)
        final_msg.setDefaultButton(QMessageBox.Ok)
        final_msg.buttonClicked.connect(self.close)
        final_msg.exec_()
        

# Final("Success", "Yah, we have done it.")