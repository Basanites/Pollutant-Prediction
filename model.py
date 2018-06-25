import pandas as pd
from util.communication import *


class Model:
    def __init__(self):
        self.events = list(['import', 'cleanup', 'filtering'])
        self.observable = Observable(self.events)
        self.df = None

    def import_csv(self, location):
        self.df = self.import_eea_weatherdata_csv(location)

    def import_csvs(self, locations):
        self.df = self.import_eea_weatherdata_csvs(locations)

    def get_concentration_series(self, station, pollutant):
        series = self.df[self.df.AirQualityStationEoICode == station][pollutant]
        return series[series.notna()]

    def get_stations(self):
        return self.df.AirQualityStationEoICode.unique()

    def get_pollutants(self):
        return self.df.columns[2:]

    def get_stations_pollutant(self, station):
        return self.df[self.df.AirQualityStationEoICode == station].dropna(1, how='all').columns[2:]

    def import_eea_weatherdata_csv(self, location: str):
        df = self._import_single_eea_weatherdata_csv(location)
        return self._tidy_up(df)

    def import_eea_weatherdata_csvs(self, locations):
        df = pd.DataFrame()

        for i, location in enumerate(locations):
            self.observable.notify('import', 'Importing File {}/{}'.format(i + 1, len(locations)))
            df = pd.concat([df, self._import_single_eea_weatherdata_csv(location)])

        return self._tidy_up(df)

    def _import_single_eea_weatherdata_csv(self, location: str):
        read = pd.read_csv(location,
                           encoding="utf-16", parse_dates=[13, 14],
                           infer_datetime_format=True,
                           index_col=[14])

        # drop 'bulk' files because they have different averaging
        bulks = read.SamplingPoint.str.lower().str.contains('bulk')
        clean = read[~bulks].copy()

        # ignore unnecessary columns
        clean.drop(columns=['Countrycode', 'Namespace', 'AirQualityNetwork',
                            'AirQualityStation', 'SamplingPoint', 'Sample',
                            'SamplingProcess', 'AirPollutantCode',
                            'DatetimeBegin', 'Validity', 'Verification',
                            'AveragingTime'],
                   inplace=True)

        return clean

    def _tidy_up(self, dataframe):
        self.observable.notify('cleanup', 'Cleaning up Dataframe')
        df = dataframe.pivot_table(columns='AirPollutant',
                                   index=[dataframe.index, 'AirQualityStationEoICode', 'UnitOfMeasurement'],
                                   values='Concentration').reset_index(level=[1, 2])

        # use shorter names
        df.index.names = ['Timestamp']
        return df.sort_index()
