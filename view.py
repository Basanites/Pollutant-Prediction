import mainwindow as m
from util.communication import *
from data.pandasmodel import PandasModel
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets
from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
from matplotlib.figure import Figure
import sys

if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)


class View:
    def __init__(self):
        # basic QtApp initialization
        self.app = QtWidgets.QApplication(sys.argv)
        self.main_window = QtWidgets.QMainWindow()
        self.ui = m.Ui_main_window()
        self.ui.setupUi(self.main_window)
        self.main_window.setWindowTitle('Predictive Analysis')
        self.files = None
        self.events = list(['tab_changed', 'combobox_changed', 'button_clicked'])
        self.observable = Observable(self.events)

        self.ui.display_layout = QtWidgets.QVBoxLayout(self.ui.display_page)
        self.ui.figure = Figure(figsize=(15, 10))
        self.ui.static_canvas = FigureCanvas(self.ui.figure)
        self.ui.toolbar = NavigationToolbar(self.ui.static_canvas, self.ui.display_page)
        self.ui.display_layout.addWidget(self.ui.static_canvas)
        self.ui.display_layout.addWidget(self.ui.toolbar)

    def run(self):
        self.main_window.show()
        sys.exit(self.app.exec_())

    def init_functionalities(self):
        self.ui.tab_widget.currentChanged.connect(lambda tab: self.observable.notify('tab_changed', tab))
        self.ui.station_combobox.currentTextChanged.connect(
            lambda text: self.observable.notify('combobox_changed', text))
        self.ui.file_select_button.clicked.connect(self.select_csvs)
        self.ui.import_button.clicked.connect(lambda: self.observable.notify('button_clicked', 'import'))
        self.ui.line_edit.textEdited.connect(self.set_files)
        self.ui.plot_button.clicked.connect(lambda: self.observable.notify('button_clicked', 'plot'))

    def select_csvs(self):
        dlg = QtWidgets.QFileDialog(caption='Open CSVs', filter='*.csv')
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)

        if dlg.exec_():
            self.set_files(dlg.selectedFiles())
            self.ui.line_edit.setText(str(self.files))

    def set_files(self, new_files):
        self.files = new_files

    #def switch_stacked_view(self, index):
    #    self.ui.stackedWidget.setCurrentIndex(index)

    def update_db_view(self, dataframe):
        self.update_statusbar('Loading Table View')
        self.ui.table_view.setModel(PandasModel(dataframe))
        self.update_statusbar('')

    def get_plot_settings(self):
        return {'station': self.ui.station_combobox.currentText(),
                'pollutant': self.ui.pollutant_combobox.currentText()}

    def update_plot_canvas(self, series):
        self.ui.figure.clear()
        ax = self.ui.figure.add_subplot(111)
        x = series.index
        ax.plot(x, series, '.-')
        ax.set_title('Plot')
        self.ui.static_canvas.draw()

    def update_selector_comboboxes(self, stations, pollutants):
        self.update_station_combobox(stations)
        self.update_pollutant_combobox(pollutants)
        self.ui.station_combobox.currentTextChanged.emit(self.ui.station_combobox.currentText())

    def update_pollutant_combobox(self, pollutants):
        self.ui.pollutant_combobox.clear()
        self.ui.pollutant_combobox.addItems(pollutants)

    def update_station_combobox(self, stations):
        self.ui.station_combobox.clear()
        self.ui.station_combobox.addItems(stations)

    def update_statusbar(self, text):
        self.ui.statusbar.showMessage(text)
