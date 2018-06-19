import model as m
import view as v


class Controller:
    def __init__(self):
        self.model = m.Model()
        self.view = v.View()
        self.view.init_functionalities(load_callback=self.load_csv)

    def run(self):
        self.view.run()

    def load_csv(self, location):
        if location:
            self.model.import_csvs(location)
            self.view.update_db_view(self.model.df)