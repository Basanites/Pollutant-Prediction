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

testing = True

if not os.path.exists(modeldir):
    os.makedirs(modeldir)
    models = list()
else:
    models = glob.glob(modeldir + '/*')

if not os.path.isfile(statsfile):
    stats_exists = False
else:
    stats_exists = True


def multiforecast(model, station, pollutant, frequency):
    """
    Runs all possible forecasts using the given parameters
    :param model:       the model to use
    :param station:     the station to predict for
    :param pollutant:   the pollutant to predict
    :return:
    """
    stats = dict((forecast_type, list()) for forecast_type in forecast_types)
    values = dict((forecast_type, list()) for forecast_type in forecast_types)

    def create_filename(station, pollutant, forecast_type, mode, parameter):
        namestring = f'{station}-{pollutant}-{forecast_type}'
        if forecast_type == 'random_forest':
            namestring = namestring + f'-estimators={parameter}'
        elif forecast_type == 'decision_tree':
            namestring = namestring + f'-depth={parameter}'
        elif forecast_type == 'knn':
            namestring = namestring + f'-neighbors={parameter}'
        namestring += f'-{mode}'
        filelocation = f'{modeldir}/{namestring}.pkl'

        return (namestring, filelocation)

    def use_model(model, callback, namestring, filelocation, forecast_type, statsdict):
        info = namestring.split('-')

        if not filelocation in models:
            print(f'running forecast for {info[0]} on {info[1]} using {info[2]} and {info[3]}')
            values[forecast_type] = callback()

            pickle.dump(model, open(filelocation, 'wb'))
        else:
            print(f'model {namestring} already created, using old stats')
            model = pickle.load(open(filelocation, 'rb'))

        statsdict[forecast_type].append(model.predictor.get_prediction_stats())

    for estimators in estimatornums:
        for mode in modes:
            filename = create_filename(station, pollutant, 'random_forest', mode, estimators)

            def callback():
                model.forecast_series(station=station, pollutant=pollutant, forecast_type='random_forest',
                                      rforest_estimators=estimators, multistepmode=mode, steps=10, frequency=frequency)

            use_model(model, callback, filename[0], filename[1], 'random_forest', stats)

    for depth in depthnums:
        for mode in modes:
            filename = create_filename(station, pollutant, 'decision_tree', mode, depth)

            def callback():
                model.forecast_series(station=station, pollutant=pollutant, forecast_type='decision_tree',
                                      dtree_depth=depth, multistepmode=mode, steps=10, frequency=frequency)

            use_model(model, callback, filename[0], filename[1], 'decision_tree', stats)

    for neighbors in neighbornums:
        for mode in modes:
            filename = create_filename(station, pollutant, 'knn', mode, neighbors)

            def callback():
                model.forecast_series(station=station, pollutant=pollutant, forecast_type='knn',
                                      knn_neighbors=neighbors, multistepmode=mode, steps=10, frequency=frequency)

            use_model(model, callback, filename[0], filename[1], 'knn', stats)

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

        if testing:
            df = df.iloc[:200]

        rate = df.iloc[0]['AveragingTime']
        delta = get_timeframe(df.index)

        if ((rate == 'day') and (len(df) >= keep_threshold * delta.days)) or (
                (rate == 'hour') and (len(df) >= keep_threshold * delta.days * 24)):

            df = df.resample(pandasrates[rate]).bfill(limit=1).interpolate(method='time')
            model = m.Model(df)

            comparison = multiforecast(model=model, station=station, pollutant=pollutant, frequency=pandasrates[rate])

            statsdf = pd.DataFrame()

            for forecast_type in comparison[0].keys():
                for stats in comparison[0][forecast_type]: ## TODO should be compressable into single for loop
                    current_frame = pd.DataFrame.from_records([stats])
                    current_frame['forecast_type'] = forecast_type
                    statsdf = pd.concat([statsdf, current_frame], ignore_index=True)

            statsdf['station'] = station
            statsdf = statsdf.set_index('station')

            #### TODO possible problem when reading data and head changes for data used right now
            if not stats_exists:
                statsdf.to_csv(statsfile)
                stats_exists = True
            else:
                with open(statsfile, 'a') as f:
                    statsdf.to_csv(f, header=False)
        else:
            print(
                f'less than {keep_threshold * 100}% of values measured for specified rate, skipping {csv}\ndelta={delta} rate={rate}')

#TODO use frequency as input, so daily data gets shifted correctly