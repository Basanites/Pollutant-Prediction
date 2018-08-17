import glob

import numpy as np
import pandas as pd
from sklearn import neighbors, ensemble, tree, linear_model
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras import Sequential, optimizers
from tensorflow.python.keras.layers import GRU, Dropout, Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

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


def create_gru(weights, input_shape, dropout_rate, learning_rate):
    optimizer = optimizers.RMSprop(lr=learning_rate)

    model = Sequential()
    model.add(GRU(weights, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer, loss='mse')
    return model


def scale_series(series, scaler=None, feature_range=(-1, 1)):
    x = series.values
    x = x.reshape(len(x), 1)

    if scaler:
        scaled = scaler.transform(x)
    else:
        scaler = MinMaxScaler(feature_range=feature_range)
        scaler = scaler.fit(x)
        scaled = scaler.transform(x)

    scaled = scaled.reshape(len(scaled))
    scaled_series = pd.Series(data=scaled, index=series.index)
    return scaled_series, scaler


def scale_dataframe(dataframe, feature_range=(-1, 1)):
    x = dataframe.values
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler = scaler.fit(x)
    scaled = scaler.transform(x)
    scaled_dataframe = pd.DataFrame(scaled, columns=dataframe.columns, index=dataframe.index)
    return scaled_dataframe, scaler


def rescale_series(series, scaler):
    x = series.values
    x = x.reshape(len(x), 1)
    rescaled = scaler.inverse_transform(x)
    rescaled = rescaled.reshape(len(rescaled))
    rescaled_series = pd.Series(data=rescaled, index=series.index)
    return rescaled_series


def rescale_dataframe(dataframe, scaler):
    x = dataframe.values
    rescaled = scaler.inverse_transform(x)
    rescaled_dataframe = pd.DataFrame(rescaled, columns=dataframe.columns, index=dataframe.index)
    return rescaled_dataframe


def scale_inputs(x, y):
    scaled_x, x_scaler = scale_dataframe(x)
    scaled_y, y_scaler = scale_series(y)
    return scaled_x, scaled_y, x_scaler, y_scaler


def rescale_input(x, y, scaler):
    rescaled_x = rescale_dataframe(x, scaler)
    rescaled_y = rescale_series(y, scaler)
    return rescaled_x, rescaled_y


def estimate_knn(x, y):
    knn = RandomizedSearchCV(neighbors.KNeighborsRegressor(),
                             param_distributions={
                                 'n_neighbors': range(2, 50 + 1, 2),
                                 'weights': ['uniform', 'distance']
                             },
                             n_iter=20,
                             n_jobs=-1)
    knn.fit(x, y)
    print(knn.best_params_, '\n', knn.best_score_)


def estimate_decistion_tree(x, y):
    decision_tree = GridSearchCV(tree.DecisionTreeRegressor(),
                                 param_grid={
                                     'max_depth': range(3, 25 + 1, 2)
                                 },
                                 n_jobs=-1)
    decision_tree.fit(x, y)
    print(decision_tree.best_params_, '\n', decision_tree.best_score_)


def estimate_random_forest(x, y):
    random_forest = RandomizedSearchCV(ensemble.RandomForestRegressor(),
                                       param_distributions={
                                           'n_estimators': range(5, 125 + 1, 5),
                                           # 'max_depth': [None, 5, 10, 20],
                                       },
                                       n_iter=20,
                                       n_jobs=-1)
    random_forest.fit(x, y)
    print(random_forest.best_params_, '\n', random_forest.best_score_)


def estimate_linear_regression(x, y):
    linear_regression = GridSearchCV(linear_model.LinearRegression(),
                                     param_grid={
                                         'normalize': [True, False]
                                     })
    linear_regression.fit(x, y)
    print(linear_regression.best_params_, '\n', linear_regression.best_score_)


def estimate_gru(x, y, rate):
    x, y, x_scaler, y_scaler = scale_inputs(x, y)
    x = x.values.reshape(x.shape[0], x.shape[1], 1)
    y = y.values
    batch_size = [24, 24 * 7] if rate == 'H' else [7, 7 * 30]

    gru = RandomizedSearchCV(KerasRegressor(create_gru, verbose=0),
                             param_distributions={
                                 'weights': np.linspace(1, 100, 20, endpoint=True, dtype=int),
                                 'dropout_rate': np.linspace(0.1, 0.3, 3, endpoint=True),
                                 'input_shape': [(x.shape[1], x.shape[2])],
                                 'epochs': range(1, 10 + 1),
                                 'batch_size': batch_size,
                                 'learning_rate': np.linspace(0.001, 0.02, 10, endpoint=True)
                             },
                             verbose=2,
                             n_iter=20,
                             n_jobs=-1)
    gru.fit(x, y)
    print(gru.best_params_, '\n', gru.best_score_)


def parameter_estimation(x, y, rate):
    # estimate_knn(x, y)
    # estimate_decistion_tree(x, y)
    # estimate_random_forest(x, y)
    # estimate_linear_regression(x, y)
    estimate_gru(x, y, rate)


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

        parameter_estimation(artificial[:-i], rotated, rate)
        if len(rest.columns.tolist()) > 1:
            parameter_estimation(rest[:-i], rotated, rate)


def difference_series(series, stepsize=1):
    """
    Differentiates a Pandas Series object.
    The returned Series does not contain the first index of the input, because there is now sensible value for it.

    :param series:      The series to differentiate
    :param stepsize:    The stepsize to use for differentiation
    :return:            The differentiated Series
    """
    diff = list()
    for i in range(stepsize, len(series), stepsize):
        diff.append(series.iloc[i] - series.iloc[i - stepsize])
    out = pd.Series(diff)
    out.index = series.index[stepsize::stepsize]
    return out


def dedifference_series(series, start_level, stepsize=1):
    """
    Inverse transforms a differentiated Pandas Series Object.
    The start_level needs to be the original series' first index to get the original series.
    The inversion works via summation from the start_level.

    :param series:      The differentiated series
    :param start_level: The start level to use for the inversion.
    :param stepsize:    The stepsize used for differentiation
    :return:            The dedifferentiated series
    """
    dediff = list()
    sum = start_level
    for i in range(0, len(series), stepsize):
        sum += series.iloc[i]
        dediff.append(sum)
    out = pd.Series(dediff)
    out.index = series.index
    return out


def test_pollutants(dataframe, rate):
    """
    Tests everything for all pollutants in a dataframe

    :param dataframe:   The dataframe to test
    :param rate:        The rate of the samples ('H' or 'D')
    """
    for pollutant in dataframe.columns:
        model_testing(dataframe, pollutant, rate)
        model_testing()


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
