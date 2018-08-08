import model as m
import glob
import os
import pandas as pd
import pickle

datadir = './post'
modeldir = './models'
statsfile = './stats.csv'
files = glob.glob(datadir + '/*')

forecast_types = ['random_forest', 'decision_tree', 'knn']

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


for csv in files:
    info = csv.replace(f'{datadir}/', '').replace('.csv', '').split('-')
    station = info[0]
    pollutant = '-'.join(info[1:])
    model = m.Model(pd.read_csv(csv, index_col=0, parse_dates=[0], infer_datetime_format=True))

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
