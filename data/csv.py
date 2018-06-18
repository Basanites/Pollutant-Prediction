import pandas as pd


def import_eea_weatherdata_csv(location: str):
    read = pd.read_csv(location,
                       encoding="utf-16", parse_dates=[13, 14],
                       infer_datetime_format=True,
                       index_col=[4, 14])

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

    # make pollutant a column for better memory usage
    clean = clean.pivot(columns='AirPollutant')

    # use shorter names
    clean.index.names = ['StationEoI', 'Timestamp']
    clean.columns.names = [None, 'Pollutant']
    return clean.sort_index()


def import_eea_weatherdata_csvs(locations):
    df = pd.DataFrame()

    for location in locations:
        df = pd.concat([df, import_eea_weatherdata_csv(location)])

    df = df.sort_index()
    return df.groupby(level=[0, 1]).first().flatten()
