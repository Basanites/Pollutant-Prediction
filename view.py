import mainwindow as m
from util.communication import *
from data.pandasmodel import PandasModel
from PyQt5 import QtWidgets
from matplotlib.backends.qt_compat import QtWidgets, is_pyqt5
from matplotlib.figure import Figure
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
import numpy as np
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
        self.events = list(['tab_changed', 'combobox_changed', 'button_clicked', 'plotting', 'finished'])
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

    def update_db_view(self, dataframe):
        self.observable.notify('plotting', 'Plotting table view')
        self.ui.table_view.setModel(PandasModel(dataframe))
        self.observable.notify('finished', 'Finished plotting table view')

    def get_plot_settings(self):
        return {'station': self.ui.station_combobox.currentText(),
                'pollutant': self.ui.pollutant_combobox.currentText()}

    def update_plot_canvas(self, series):
        series = series.asfreq(freq='1H', fill_value=np.nan).interpolate(method='time')
        self.observable.notify('plotting', 'Plotting data')
        self.ui.figure.clear()
        ax = self.ui.figure.add_subplot(611)
        x = series.index
        ax.plot(x, series, '-')
        ax.set_title('observed', rotation='vertical',x=-0.1,y=0.5)
        decomp = seasonal_decompose(series, model='additive', freq=24*7)
        ax2 = self.ui.figure.add_subplot(612)
        ax2.plot(x, decomp.seasonal, '-')
        ax2.set_title('seasonal', rotation='vertical',x=-0.1,y=0.5)
        ax3 = self.ui.figure.add_subplot(613)
        ax3.plot(x, decomp.trend, '-')
        ax3.set_title('trend', rotation='vertical',x=-0.1,y=0.5)
        ax4 = self.ui.figure.add_subplot(614)
        ax4.plot(x, decomp.resid, '-')
        ax4.set_title('residual', rotation='vertical',x=-0.1,y=0.5)
        autocorr = acf(series, nlags=24, alpha=0.05)
        ax5 = self.ui.figure.add_subplot(615)
        ax5.plot(range(0, 25), autocorr[0], '.-')
        #ax5.plot(range(0, 25), autocorr[1], '-', color='r')
        ax5.set_title('acf', rotation='vertical',x=-0.1,y=0.5)
        pautocorr = pacf(series, nlags=24, alpha=0.05)
        ax6 = self.ui.figure.add_subplot(616)
        ax6.plot(range(0, 25), pautocorr[0], '.-')
        #ax6.plot(range(0,25), pautocorr[1], '-', color='r')
        ax6.set_title('pacf', rotation='vertical',x=-0.1,y=0.5)


        self.ui.static_canvas.draw()
        self.observable.notify('finished', 'Finished plotting selected data')

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

    def update_statusbar(self, text, msecs=0):
        self.ui.statusbar.showMessage(text, msecs)
