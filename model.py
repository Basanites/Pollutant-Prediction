from data.csv import import_eea_weatherdata_csv, import_eea_weatherdata_csvs


class Model:
    def __init__(self):
        self.df = None

    def import_csv(self, location):
        self.df = import_eea_weatherdata_csv(location)

    def import_csvs(self, locations):
        self.df = import_eea_weatherdata_csvs(locations)

    def get_stations(self):
        return self.df.AirQualityStationEoICode.unique()

    def get_pollutants(self):
        return self.df.columns[2:]

    def get_stations_pollutant(self, station):
        return self.df[self.df.AirQualityStationEoICode == station].dropna(1, how='all').columns[2:]