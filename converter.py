import glob
import os
import sys

import pandas as pd

datadir = './res'
finaldir = './post'
keep_threshold = 0.95

pre = glob.glob(datadir + '/*')
if not os.path.exists(finaldir):
    os.makedirs(finaldir)
    post = list()
else:
    post = glob.glob(finaldir + '/*')


def _import_single_eea_weatherdata_csv(location: str):
    """
    Returns a pandas df with only the relevant columns from a specified eea csv file

    :param location:    The location of the original csv
    :return:            The according dataframe
    """
    read = _read_whole_csv(location)
    clean = _drop_unneccessary_entries(read)
    return _drop_unneccessary_columns(clean)


def _read_whole_csv(location):
    """
    Returns the whole df from an eea csv

    :param location:    The location of the csv
    :return:            The according dataframe
    """
    return pd.read_csv(location,
                       encoding="utf-16", parse_dates=[13, 14],
                       infer_datetime_format=True,
                       index_col=[14])


def _drop_unneccessary_entries(df):
    """
    Removes all bulk entries, because they don't match the averaging

    :param df:  The dataframe to remove unneccessary entries from
    :return:    The filtered dataframe
    """
    bulks = df.SamplingPoint.str.lower().str.contains('bulk')
    return df[~bulks].copy()


def _drop_unneccessary_columns(df):
    """
    Removes all unneeded columns from an eea dataframe

    :param df:  The dataframe to filter
    :return:    The filtered dataframe
    """
    return df.drop(columns=['Countrycode', 'Namespace', 'AirQualityNetwork',
                            'AirQualityStation', 'SamplingPoint', 'Sample',
                            'SamplingProcess', 'AirPollutantCode', 'UnitOfMeasurement',
                            'DatetimeBegin', 'Validity', 'Verification'])


def _tidy_up(df):
    """
    Makes an eea dataframe a bit more accessible

    :param df:  The dataframe to transform
    :return:    The transformed dataframe
    """
    df = _descriptors_as_columns(df)
    _set_short_names(df)
    return df.sort_index()


def _descriptors_as_columns(df):
    """
    Makes the pollutants columns with their according concentration as values

    :param df:  The eea dataframe
    :return:    The modified dataframe
    """
    return df.pivot_table(columns='AirPollutant',
                          index=[df.index, 'AirQualityStationEoICode', 'AveragingTime'],
                          values='Concentration').reset_index(level=[1, 2])


def _set_short_names(df):
    """
    Sets shorter names for the eea dataframe columns

    :param df:  The dataframe to modify
    :return:    The modified dataframe
    """
    df.index.names = ['Timestamp']


def get_timeframe(index):
    """
    Calculates the time delta between the start and end of a pandas index

    :param index:   the pandas index
    :return:        a datetime timedelta object
    """
    delta = index[-1].to_pydatetime() - index[0].to_pydatetime()
    return delta


def convert():
    """
    Converts the eea csvs to new system
    """
    i = 0
    for csv in pre:
        i += 1

        df = _import_single_eea_weatherdata_csv(csv)
        try:
            station = df.iloc[0].AirQualityStationEoICode
            averaging = df.iloc[0].AveragingTime

            filename = f'{station}-{averaging}.csv'
            filelocation = f'{finaldir}/{filename}'

            sys.stdout.write(f'\r{i}/{len(pre)}\t\treading {csv}')

            if filelocation in post:
                df2 = pd.read_csv(filelocation, index_col=0, parse_dates=[0], infer_datetime_format=True)
                df2 = df2.drop(columns=['AirQualityStationEoICode', 'AveragingTime'])

                df = df.combine_first(df2)
                df['AirQualityStationEoICode'] = station
                df['AveragingTime'] = averaging
                df = df.sort_index()
            else:
                post.append(filelocation)
            df.to_csv(filelocation)
        # Index error means the df is broken, skip these
        except IndexError:
            pass


def remove_unneccessary():
    """
    Removes unneccessary pollutants / dataframes if the sampling density threshold is not reached
    """
    for csv in post:
        df = pd.read_csv(csv, index_col=0, parse_dates=[0], infer_datetime_format=True)

        rate = df.iloc[0]['AveragingTime']
        delta = get_timeframe(df.index)

        columns = df.columns.drop(['AveragingTime', 'AirQualityStationEoICode'])

        for column in columns:
            if not (((rate == 'day') and (len(df[column].dropna()) >= keep_threshold * delta.days)) or (
                    (rate == 'hour') and (len(df[column].dropna()) >= keep_threshold * delta.days * 24))):
                print(f'removing {column} from {csv} because of bad resolution')
                df = df.drop(columns=[column])

        if not len(df.columns.drop(['AveragingTime', 'AirQualityStationEoICode'])):
            print(f'removing {csv} because it has no entries anymore')
            os.remove(csv)
        else:
            df.to_csv(csv)


if __name__ == '__main__':
    convert()
    remove_unneccessary()
