from data.csv import import_eea_weatherdata_csv, import_eea_weatherdata_csvs


class Model:
    def __init__(self):
        self.df = None

    def import_csv(self, location):
        self.df = import_eea_weatherdata_csv(location)

    def import_csvs(self, locations):
        self.df = import_eea_weatherdata_csvs(locations)