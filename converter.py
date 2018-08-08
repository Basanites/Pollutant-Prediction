import glob
import os
from model import Model
import pandas as pd
import sys

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
            columns = list(df)
            station = df.iloc[0].AirQualityStationEoICode

            filename = f'{station}-{columns[-1]}.csv'
            filelocation = f'{finaldir}/{filename}'

            sys.stdout.write(f'\r{i}/{len(pre)}\t\treading {csv}')

            if filelocation in post:
                df2 = pd.read_csv(filelocation, index_col=0, parse_dates=[0], infer_datetime_format=True)
                df = pd.concat([df, df2])
                df = df[~df.index.duplicated(keep='first')]
                df = df.sort_index()
            else:
                post.append(filelocation)
            df.to_csv(filelocation)
        except IndexError:
            pass


def remove_unneccessary():
    for csv in post:
        info = csv.replace(f'{datadir}/', '').replace('.csv', '').split('-')
        df = pd.read_csv(csv, index_col=0, parse_dates=[0], infer_datetime_format=True)

        rate = df.iloc[0]['AveragingTime']
        delta = get_timeframe(df.index)

        check1 = (rate == 'day') and (len(df) >= keep_threshold * delta.days)
        check2 = (rate == 'day') and (len(df) >= keep_threshold * delta.days)

        if ((rate == 'day') and (len(df) >= keep_threshold * delta.days)) or (
                (rate == 'hour') and (len(df) >= keep_threshold * delta.days * 24)):
            pass
        else:
            print(f'removing {csv}, rate=={rate}, days=={delta.days}, hours=={delta.days * 24}, length={len(df)}, {check1}, {check2}')
            os.remove(csv)


if __name__ == '__main__':
    convert()
    remove_unneccessary()
