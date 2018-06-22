import mainwindow as m
import qtpandas.models.DataFrameModel as dm
from PyQt5 import QtWidgets
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

    def init_functionalities(self, load_callback, tab_callback):
        # tabs
        self.ui.tab_widget.currentChanged.connect(tab_callback)

        # statistics_stack
        # prediction_stack

        # buttons
        self.ui.file_select_button.clicked.connect(self.select_csvs)
        self.ui.import_button.clicked.connect(lambda: load_callback(self.files))

        # line edit
        self.ui.line_edit.textEdited.connect(self.set_files)

    def select_csvs(self):
        dlg = QtWidgets.QFileDialog(caption='Open CSVs', filter='*.csv')
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)

        if dlg.exec_():
            self.set_files(dlg.selectedFiles())
            self.ui.line_edit.setText(str(self.files))

    def set_files(self, new_files):
        self.files = new_files

    def update_db_view(self, dataframe):
        self.ui.tableView.setModel(dm.DataFrameModel(dataframe))
