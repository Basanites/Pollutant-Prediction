import mainwindow as m
from PyQt5 import QtWidgets
import sys


class View:
    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.main_window = QtWidgets.QMainWindow()
        self.ui = m.Ui_main_window()
        self.ui.setupUi(self.main_window)
        self.main_window.setWindowTitle('Predictive Analysis')

    def run(self):
        self.main_window.show()
        sys.exit(self.app.exec_())
