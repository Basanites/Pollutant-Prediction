import model as m
import view as v
from util.communication import *


class Controller(Observer):
    def __init__(self):
        super(Controller, self).__init__()
        self.model = m.Model()
        self.view = v.View()
        self.view.init_functionalities()
        self.view.observable.register(self, 'tab_changed', self.change_tab)
        self.view.observable.register(self, 'combobox_changed', self.change_combobox)
        self.view.observable.register(self, 'button_clicked', self.click_button)
        self.model.observable.register(self, 'import', self.view.update_statusbar)
        self.model.observable.register(self, 'cleanup', self.view.update_statusbar)
        self.tab_needs_update = [False] * 4

    def run(self):
        self.view.run()

    def load_csvs(self, locations):
        if locations:
            self.model.import_csvs(locations)
        self.view.update_statusbar('')

    def click_button(self, button_id):
        if button_id == 'import':
            self.load_csvs(self.view.files)
            self.tab_needs_update[1:3] = [True] * 2
        elif button_id == 'plot':
            settings = self.view.get_plot_settings()
            self.view.update_plot_canvas(
                self.model.get_concentration_series(settings['station'], settings['pollutant']))

    def change_tab(self, tab_id):
        if tab_id == 1 and self.model.df is not None and self.tab_needs_update[1]:
            self._update_display_tab()
            self.tab_needs_update[1] = False
        elif tab_id == 2 and self.model.df is not None and self.tab_needs_update[2]:
            self.view.update_selector_comboboxes(self.model.get_stations(), self.model.get_pollutants())
            self.tab_needs_update[2] = False

    def _update_display_tab(self):
        df = self.model.df.reset_index()
        df['Timestamp'] = df['Timestamp'].apply(str)
        self.view.update_db_view(df)

    def change_combobox(self, new_text):
        self.view.update_pollutant_combobox(self.model.get_stations_pollutant(new_text))
