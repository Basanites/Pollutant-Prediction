import model as m
import timeseries.predictions
import glob
import os
import pandas as pd
import pickle

datadir = './post'
modeldir = './models'
statsfile = './stats.csv'
files = glob.glob(datadir)

forecast_types = ['random_forest', 'decision_tree', 'knn']

if not os.path.exists(modeldir):
    os.makedirs(modeldir)
    models = list()
else:
    models = glob.glob(modeldir+ '/*')

if not os.isfile(statsfile):
    stats_exists = False
else:
    stats_exists = True

def multiforecast(model, station, pollutant):
    stats = dict()
    values = dict()

    for forecast_type in forecast_types:
        namestring = f'{station}-{pollutant}-{forecast_type}'
        filelocation = f'{modeldir}/{namestring}.pkl'

        values[forecast_type] = model.forecast_series(station, pollutant)

        pickle.dump(model, filelocation)

        stats[forecast_type] = model.predictors[0].get_prediction_stats()

    return (stats, values)

for pkl in files:
    info = pkl.replace(f'{datadir}/', '').replace('.pkl', '').split('-')
    station, pollutant = info[0], info[1]
    model = m.Model(pd.read_pickle(pkl))

    comparison = multiforecast(model, station, pollutant)

    statsdf = pd.DataFrame()

    for forecast_type in forecast_types:
        statsdf[forecast_type] = pd.DataFrame.from_dict(comparison[0][forecast_type])

    statsdf['station'] = pd.Series(station, index=statsdf.index)

    statsdf.reindex('station')

    with open('foo.csv', 'a') as f:
        if stats_exists:
            statsdf.to_csv(statsfile)
        else:
            statsdf.to_csv(statsfile, header=False)