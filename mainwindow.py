# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_main_window(object):
    def setupUi(self, main_window):
        main_window.setObjectName("main_window")
        main_window.resize(333, 167)
        main_window.setMinimumSize(QtCore.QSize(333, 167))
        main_window.setTabShape(QtWidgets.QTabWidget.Rounded)
        main_window.setDockNestingEnabled(False)
        self.centralwidget = QtWidgets.QWidget(main_window)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.tab_widget = QtWidgets.QTabWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tab_widget.sizePolicy().hasHeightForWidth())
        self.tab_widget.setSizePolicy(sizePolicy)
        self.tab_widget.setTabPosition(QtWidgets.QTabWidget.North)
        self.tab_widget.setObjectName("tab_widget")
        self.import_tab = QtWidgets.QWidget()
        self.import_tab.setObjectName("import_tab")
        self.gridLayout = QtWidgets.QGridLayout(self.import_tab)
        self.gridLayout.setObjectName("gridLayout")
        self.widget = QtWidgets.QWidget(self.import_tab)
        self.widget.setMaximumSize(QtCore.QSize(16777215, 50))
        self.widget.setObjectName("widget")
        self.formLayout = QtWidgets.QFormLayout(self.widget)
        self.formLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.formLayout.setObjectName("formLayout")
        self.file_select_button = QtWidgets.QPushButton(self.widget)
        self.file_select_button.setObjectName("file_select_button")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.file_select_button)
        self.line_edit = QtWidgets.QLineEdit(self.widget)
        self.line_edit.setObjectName("line_edit")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.line_edit)
        self.gridLayout.addWidget(self.widget, 0, 0, 1, 1)
        self.import_button = QtWidgets.QPushButton(self.import_tab)
        self.import_button.setObjectName("import_button")
        self.gridLayout.addWidget(self.import_button, 1, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 2, 0, 1, 1)
        self.tab_widget.addTab(self.import_tab, "")
        self.display_tab = QtWidgets.QWidget()
        self.display_tab.setObjectName("display_tab")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.display_tab)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.text_browser = QtWidgets.QTextBrowser(self.display_tab)
        self.text_browser.setObjectName("text_browser")
        self.gridLayout_2.addWidget(self.text_browser, 0, 0, 1, 1)
        self.tab_widget.addTab(self.display_tab, "")
        self.statistics_tab = QtWidgets.QWidget()
        self.statistics_tab.setObjectName("statistics_tab")
        self.tab_widget.addTab(self.statistics_tab, "")
        self.predictions_tab = QtWidgets.QWidget()
        self.predictions_tab.setObjectName("predictions_tab")
        self.tab_widget.addTab(self.predictions_tab, "")
        self.gridLayout_3.addWidget(self.tab_widget, 0, 0, 1, 1)
        main_window.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(main_window)
        self.statusbar.setObjectName("statusbar")
        main_window.setStatusBar(self.statusbar)

        self.retranslateUi(main_window)
        self.tab_widget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(main_window)

    def retranslateUi(self, main_window):
        _translate = QtCore.QCoreApplication.translate
        main_window.setWindowTitle(_translate("main_window", "MainWindow"))
        self.file_select_button.setText(_translate("main_window", "Select File"))
        self.import_button.setText(_translate("main_window", "Import"))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.import_tab), _translate("main_window", "Import"))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.display_tab), _translate("main_window", "Display"))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.statistics_tab), _translate("main_window", "Statistics"))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.predictions_tab), _translate("main_window", "Predictions"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    main_window = QtWidgets.QMainWindow()
    ui = Ui_main_window()
    ui.setupUi(main_window)
    main_window.show()
    sys.exit(app.exec_())

