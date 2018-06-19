import model as m
import view as v


class Controller:
    def __init__(self):
        self.model = m.Model()
        self.view = v.View()

    def run(self):
        self.view.run()

    def import_csv(self, location):
        self.model.import_csv(location)
