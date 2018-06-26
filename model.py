import pandas as pd
from util.communication import *


class Model:
    def __init__(self):
        self.events = list(['import', 'cleanup', 'filtering', 'finished'])
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

        self.observable.notify('finished', 'Finished loading')
        return self._tidy_up(df)

    def _import_single_eea_weatherdata_csv(self, location: str):
        read = self._read_whole_csv(location)
        clean = self._drop_unneccessary_entries(read)
        return self._drop_unneccessary_columns(clean)

    def _read_whole_csv(self, location):
        return pd.read_csv(location,
                           encoding="utf-16", parse_dates=[13, 14],
                           infer_datetime_format=True,
                           index_col=[14])

    def _drop_unneccessary_entries(self, df):
        bulks = df.SamplingPoint.str.lower().str.contains('bulk')
        return df[~bulks].copy()

    def _drop_unneccessary_columns(self, df):
        return df.drop(columns=['Countrycode', 'Namespace', 'AirQualityNetwork',
                            'AirQualityStation', 'SamplingPoint', 'Sample',
                            'SamplingProcess', 'AirPollutantCode',
                            'DatetimeBegin', 'Validity', 'Verification',
                            'AveragingTime'],
                   inplace=True)

    def _tidy_up(self, df):
        self.observable.notify('cleanup', 'Cleaning up Dataframe')
        df = self._descriptors_as_columns(df)
        self._set_short_names(df)
        self.observable.notify('finished', 'Finished cleaning up data')
        return df.sort_index()

    def _descriptors_as_columns(self, df):
        return df.pivot_table(columns='AirPollutant',
                                   index=[df.index, 'AirQualityStationEoICode', 'UnitOfMeasurement'],
                                   values='Concentration').reset_index(level=[1, 2])

    def _set_short_names(self, df):
        df.index.names = ['Timestamp']

