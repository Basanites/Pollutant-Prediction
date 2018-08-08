import model as m
import glob
import os
import pandas as pd
import pickle
import datetime

datadir = './post'
modeldir = './models'
statsfile = './stats.csv'
files = glob.glob(datadir + '/*')
keep_threshold = 0.95
pandasrates = {'day': 'D', 'hour': 'H'}

forecast_types = ['random_forest', 'decision_tree', 'knn']

if not os.path.exists(modeldir):
    os.makedirs(modeldir)
    models = list()
else:
    models = glob.glob(modeldir + '/*')


def multiforecast(model, station, pollutant):
    """
    Runs all possible forecasts using the given parameters
    :param model:       the model to use
    :param station:     the station to predict for
    :param pollutant:   the pollutant to predict
    :return:
    """
    stats = dict()
    values = dict()

    for forecast_type in forecast_types:
        namestring = f'{station}-{pollutant}-{forecast_type}'
        filelocation = f'{modeldir}/{namestring}.pkl'

        if not filelocation in models:
            print(f'running forecast for {station} on {pollutant} using {forecast_type}')
            values[forecast_type] = model.forecast_series(station, pollutant)

            pickle.dump(model, open(filelocation, 'wb'))
        else:
            print(f'model {namestring} already created, using old stats')
            model = pickle.load(open(filelocation, 'rb'))

        stats[forecast_type] = model.predictor.get_prediction_stats()

    return (stats, values)


def get_timeframe(index):
    """
    Calculates the time delta between the start and end of a pandas index
    :param index:   the pandas index
    :return:        a datetime timedelta object
    """
    delta = index[-1].to_pydatetime() - index[0].to_pydatetime()
    return delta


def main():
    """
    Main program
    """
    if not os.path.isfile(statsfile):
        stats_exists = False
    else:
        stats_exists = True

    for csv in files:
        info = csv.replace(f'{datadir}/', '').replace('.csv', '').split('-')
        station = info[0]
        pollutant = '-'.join(info[1:])
        df = pd.read_csv(csv, index_col=0, parse_dates=[0], infer_datetime_format=True)

        rate = df.iloc[0]['AveragingTime']
        delta = get_timeframe(df.index)

        if ((rate == 'day') and (len(df) >= keep_threshold * delta.days)) or (
                (rate == 'hour') and (len(df) >= keep_threshold * delta.days * 24)):

            df = df.resample(pandasrates[rate]).interpolate(method='time')
            model = m.Model(df)

            comparison = multiforecast(model, station, pollutant)

            statsdf = pd.DataFrame()

            for forecast_type in comparison[0].keys():
                current_frame = pd.DataFrame.from_records([comparison[0][forecast_type]])
                current_frame['forecast_type'] = forecast_type
                statsdf = pd.concat([statsdf, current_frame], ignore_index=True)

            statsdf['station'] = station
            statsdf = statsdf.set_index('station')

            if not stats_exists:
                statsdf.to_csv(statsfile)
                stats_exists = True
            else:
                with open(statsfile, 'a') as f:
                    statsdf.to_csv(f, header=False)
        else:
            print(
                f'less than {keep_threshold * 100}% of values measured for specified rate, skipping {csv}\ndelta={delta} rate={rate}')


main()
