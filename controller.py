import model as m
import view as v


class Controller:
    def __init__(self):
        self.model = m.Model()
        self.view = v.View()
        self.view.init_functionalities(load_callback=self.load_csvs, tab_callback=self.change_tab)
        self.tab_needs_update = [False] * 4

    def run(self):
        self.view.run()

    def load_csvs(self, locations):
        if locations:
            self.model.import_csvs(locations)
            self.tab_needs_update[1:3] = [True] * 2

    def change_tab(self, tab_id):
        if tab_id == 1 and self.model.df is not None and self.tab_needs_update[1]:
            df = self.model.df.reset_index()
            df['Timestamp'] = df['Timestamp'].apply(str)
            self.view.update_db_view(df)
            self.tab_needs_update[1] = False
        elif tab_id == 2 and self.model.df is not None and self.tab_needs_update[2]:
            self.view.update_selector_comboboxes(self.model.get_stations(), self.model.get_pollutants())
            self.tab_needs_update[2] = False
