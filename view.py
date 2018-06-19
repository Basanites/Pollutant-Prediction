import mainwindow as m
from PyQt5 import QtWidgets, QtGui
import sys


class View:
    def __init__(self):
        # basic QtApp initialization
        self.app = QtWidgets.QApplication(sys.argv)
        self.main_window = QtWidgets.QMainWindow()
        self.ui = m.Ui_main_window()
        self.ui.setupUi(self.main_window)
        self.main_window.setWindowTitle('Predictive Analysis')
        self.files = None

    def run(self):
        self.main_window.show()
        sys.exit(self.app.exec_())

    def init_functionalities(self, load_callback):
        # buttons
        self.ui.file_select_button.clicked.connect(self.load_csvs)
        self.ui.import_button.clicked.connect(lambda: load_callback(self.files))

        # line edit
        self.ui.line_edit.textEdited.connect(self.set_files)

    def load_csvs(self):
        dlg = QtWidgets.QFileDialog(caption='Open CSVs', filter='*.csv')
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)

        if dlg.exec_():
            self.set_files(dlg.selectedFiles())
            self.ui.line_edit.setText(str(self.files))

    def set_files(self, new_files):
        self.files = new_files

    def update_db_view(self, dataframe):
        self.ui.text_browser.setText(str(dataframe))
