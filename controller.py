import model as m
import view as v


class Controller:
    def __init__(self):
        self.model = m.Model()
        self.view = v.View()
        self.view.init_functionalities(load_callback=self.load_csvs, tab_callback=self.tabchanged)
        self.tab_needs_update = []

    def run(self):
        self.view.run()

    def load_csvs(self, locations):
        if locations:
            self.model.import_csvs(locations)
            self.tab_needs_update[1:2] = [True] * 2

    def tabchanged(self, tab_id):
        if tab_id == 1 and self.model.df is not None and self.tab_needs_update[1]:
            df = self.model.df.reset_index()
            df['Timestamp'] = df['Timestamp'].apply(str)
            self.view.update_db_view(df)
            self.tab_needs_update[1] = False