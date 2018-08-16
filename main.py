import glob

import pandas as pd
from sklearn import neighbors, ensemble, tree, linear_model
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from timeseries.predictions import create_artificial_features


def resample_dataframe(dataframe, rate='H'):
    if rate == 'D':
        dataframe = dataframe.resample(rate).bfill(limit=1).interpolate(
            method='time')  # bfill is used here, because daily values act up otherwise
    else:
        dataframe = dataframe.resample(rate).interpolate(method='time')
    return dataframe


def get_info(csv_path):
    info = csv_path.replace(f'{datadir}/', '').replace('.csv', '').split('-')
    station_name = info[0]
    rate = {'day': 'D', 'hour': 'H'}[info[-1]]
    return station_name, rate


def parameter_estimation(x, y):
    knn = RandomizedSearchCV(neighbors.KNeighborsRegressor(),
                             param_distributions={
                                 'n_neighbors': range(2, 50 + 1, 2),
                                 'weights': ['uniform', 'distance']
                             },
                             n_iter=20,
                             n_jobs=-1)
    knn.fit(x, y)
    print(knn.best_estimator_, '\n', knn.best_score_)

    random_forest = RandomizedSearchCV(ensemble.RandomForestRegressor(),
                                       param_distributions={
                                           'n_estimators': range(5, 125 + 1, 5),
                                           # 'max_depth': [None, 5, 10, 20],
                                       },
                                       n_iter=20,
                                       n_jobs=-1)
    random_forest.fit(x, y)
    print(random_forest.best_estimator_, '\n', random_forest.best_score_)

    decision_tree = GridSearchCV(tree.DecisionTreeRegressor(),
                                       param_grid={
                                           'max_depth': range(3, 25 + 1, 2)
                                       },
                                       n_jobs=-1)
    decision_tree.fit(x, y)
    print(decision_tree.best_estimator_, '\n', decision_tree.best_score_)

    linear_regression = GridSearchCV(linear_model.LinearRegression(),
                                           param_grid={})
    linear_regression.fit(x, y)
    print(linear_regression.best_estimator_, '\n', decision_tree.best_score_)


def rotate_series(series):
    return pd.concat([series[1:], pd.Series(series.iloc[0])])


def model_testing(dataframe, pollutant, rate):
    distance = 7 if rate == 'D' else 24  # distance for predictions, always 1 season (24 hrs or 7 days)

    series = dataframe[pollutant]
    rest = dataframe.drop(columns=[pollutant])[distance:]
    artificial = create_artificial_features(series, rate, steps=distance)[distance:]

    rotated = series[distance:]
    for i in range(1, distance + 1):
        rotated = rotate_series(rotated)[:-1]

        parameter_estimation(artificial[:-i], rotated)
        if len(rest.columns.tolist()) > 1:
            parameter_estimation(rest[:-i], rotated)


def test_pollutants(dataframe, rate):
    for pollutant in dataframe.columns:
        model_testing(dataframe, pollutant, rate)


if __name__ == '__main__':
    datadir = './post'
    modeldir = './models'
    statsfile = './stats.csv'
    files = glob.glob(datadir + '/*')

    for csv in files:
        station, steprate = get_info(csv)
        df = pd.read_csv(csv, index_col=0, parse_dates=[0], infer_datetime_format=True).drop(
            columns=['AirQualityStationEoICode', 'AveragingTime'])
        df = resample_dataframe(df, steprate)

        if len(df > 8760):
            df = df[:8760]

        test_pollutants(df, steprate)

        # stats_gru = RandomizedSearchCV()
        # stats_lstm = RandomizedSearchCV()
