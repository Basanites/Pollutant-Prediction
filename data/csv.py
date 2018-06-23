import pandas as pd


def import_eea_weatherdata_csv(location: str):
    df = _import_single_eea_weatherdata_csv(location)
    return _tidy_up(df)


def import_eea_weatherdata_csvs(locations):
    df = pd.DataFrame()

    for location in locations:
        df = pd.concat([df, _import_single_eea_weatherdata_csv(location)])

    return _tidy_up(df)


def _import_single_eea_weatherdata_csv(location: str):
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


def _tidy_up(dataframe):
    df = dataframe.pivot_table(columns='AirPollutant',
                               index=[dataframe.index, 'AirQualityStationEoICode', 'UnitOfMeasurement'],
                               values='Concentration').reset_index(level=[1, 2])

    # use shorter names
    df.index.names = ['Timestamp']
    return df.sort_index()
