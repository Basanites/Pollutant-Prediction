import glob
import os
import pickle

import pandas as pd

import model as m
from converter import get_timeframe

datadir = './post'
modeldir = './models'
statsfile = './stats.csv'
files = glob.glob(datadir + '/*')
keep_threshold = 0.95
pandasrates = {'day': 'D', 'hour': 'H'}
forecast_types = ['random_forest', 'decision_tree', 'knn']
modes = ['single', 'multimodel']
estimatornums = [10, 20, 50]
depthnums = [5, 10, 20]
neighbornums = [5, 10, 20]

if not os.path.exists(modeldir):
    os.makedirs(modeldir)
    models = list()
else:
    models = glob.glob(modeldir + '/*')

if not os.path.isfile(statsfile):
    stats_exists = False
else:
    stats_exists = True


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
    calls = {'random_forest': list(), 'decision_tree': list(), 'knn': list()}
    for estimators in estimatornums:
        for mode in modes:
            calls['random_forest'].append(
                model.forecast_series(station=station, pollutant=pollutant, rforest_estimators=estimators, mode=mode,
                                      steps=10))
    for depth in depthnums:
        for mode in modes:
            calls['decision_tree'].append(
                model.forecast_series(station=station, pollutant=pollutant, dtree_depth=depth, mode=mode, steps=10))
    for neighbors in neighbornums:
        for mode in modes:
            calls['knn'].append(
                model.forecast_series(station=station, pollutant=pollutant, knn_neighbors=neighbors, mode=mode,
                                      steps=10))

    for forecast_type, calllist in calls:
        stats[forecast_type] = list()
        for i in range(0, len(calllist)):
            namestring = f'{station}-{pollutant}-{forecast_type}'
            if forecast_type == 'random_forest':
                namestring = namestring.join(f'-estimators={estimatornums[i]}')
            elif forecast_type == 'decision_tree':
                namestring = namestring.join(f'-depth={depthnums[i]}')
            elif forecast_type == 'knn':
                namestring = namestring.join(f'-neighbors={neighbornums[i]}')
            filelocation = f'{modeldir}/{namestring}.pkl'

            info = namestring.split('-')

            if not filelocation in models:
                print(f'running forecast for {info[0]} on {info[1]} using {info[2]} and {info[3]}')
                values[forecast_type] = calllist[i]

                pickle.dump(model, open(filelocation, 'wb'))
            else:
                print(f'model {namestring} already created, using old stats')
                model = pickle.load(open(filelocation, 'rb'))

            stats[forecast_type].append(model.predictor.get_prediction_stats())

    return (stats, values)


if __name__ == '__main__':
    """
    Keeps only the series having a threshold high percentage of samples according to their specified rates.
    These are then interpolated and each possible forecast is being run.
    The prediction stats are saved in stats.csv
    """

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

            comparison = multiforecast(model=model, station=station, pollutant=pollutant)

            statsdf = pd.DataFrame()

            for forecast_type in comparison[0].keys():
                for stats in comparison[0][forecast_type]:
                    current_frame = pd.DataFrame.from_records(stats)
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
