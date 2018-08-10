import glob
import os
import sys

import pandas as pd

from model import Model

datadir = './res'
finaldir = './post'
keep_threshold = 0.95

pre = glob.glob(datadir + '/*')
if not os.path.exists(finaldir):
    os.makedirs(finaldir)
    post = list()
else:
    post = glob.glob(finaldir + '/*')


def get_timeframe(index):
    """
    Calculates the time delta between the start and end of a pandas index
    :param index:   the pandas index
    :return:        a datetime timedelta object
    """
    delta = index[-1].to_pydatetime() - index[0].to_pydatetime()
    return delta


def convert():
    model = Model()

    i = 0
    for csv in pre:
        i += 1

        model.import_csv(csv)
        df = model.df
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
