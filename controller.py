import model as m
import view as v


class Controller:
    def __init__(self):
        self.model = m.Model()
        self.view = v.View(file_callback=self.import_csv)

    def run(self):
        self.view.mainloop()

    def import_csv(self, location):
        self.model.import_csv(location)
        self.view.display_dataframe(self.model.df)
