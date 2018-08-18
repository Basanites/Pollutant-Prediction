import glob
import time

import numpy as np
import pandas as pd
from fbprophet import Prophet
from pyramid.arima import auto_arima
from sklearn import neighbors, ensemble, tree, linear_model
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tensorflow.python.keras import Sequential, optimizers
from tensorflow.python.keras.layers import GRU, Dropout, Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

from timeseries.predictions import create_artificial_features


def resample_dataframe(dataframe, rate='H'):
    """
    Resamples a dataframe according to a given rate.

    :param dataframe:   The dataframe to resample
    :param rate:        The rate to use ('D' or 'H')
    :return:            The resampled dataframe
    """
    if rate == 'D':
        dataframe = dataframe.resample(rate).bfill(limit=1).interpolate(
            method='time')  # bfill is used here, because daily values act up otherwise
    else:
        dataframe = dataframe.resample(rate).interpolate(method='time')
    return dataframe


def get_info(csv_path, directory):
    """
    Gets station name and rate from a csv path string

    :param csv_path:    The path string
    :return:            The station name and rate
    """
    info = csv_path.replace(f'{directory}/', '').replace('.csv', '').split('-')
    station_name = info[0]
    rate = {'day': 'D', 'hour': 'H'}[info[-1]]
    return station_name, rate


def create_gru(weights, input_shape, dropout_rate, learning_rate):
    """
    Creates a GRU based RNN using the given input parameters.

    :param weights:         The amount of output weights for GRU layer
    :param input_shape:     The shape of the inputs
    :param dropout_rate:    The dropout rate after GRU layer
    :param learning_rate:   The learning rate of the optimizer
    :return:                The compiled model
    """
    optimizer = optimizers.RMSprop(lr=learning_rate)

    model = Sequential()
    model.add(GRU(weights, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer, loss='mse')
    return model


def scale_series(series, feature_range=(-1, 1)):
    """
    Downscales a series to given feature_range.

    :param series:          The series to downscale
    :param feature_range:   The range of values after scaling
    :return:                The scaled series and used scaler
    """
    x = series.values
    x = x.reshape(len(x), 1)
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler = scaler.fit(x)
    scaled = scaler.transform(x)
    scaled = scaled.reshape(len(scaled))
    scaled_series = pd.Series(data=scaled, index=series.index)
    return scaled_series, scaler


def scale_dataframe(dataframe, feature_range=(-1, 1)):
    """
    Downscales all values in a dataframe to given feature_range.

    :param dataframe:       The dataframe to downscale
    :param feature_range:   The range of values after scaling
    :return:                The scaled dataframe and used scaler
    """
    x = dataframe.values
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler = scaler.fit(x)
    scaled = scaler.transform(x)
    scaled_dataframe = pd.DataFrame(scaled, columns=dataframe.columns, index=dataframe.index)
    return scaled_dataframe, scaler


def rescale_series(series, scaler):
    """
    Rescales a series using given scaler.

    :param series:  The series to rescale
    :param scaler:  The scaler used to scale the series
    :return:        The rescaled series
    """
    x = series.values
    x = x.reshape(len(x), 1)
    rescaled = scaler.inverse_transform(x)
    rescaled = rescaled.reshape(len(rescaled))
    rescaled_series = pd.Series(data=rescaled, index=series.index)
    return rescaled_series


def rescale_dataframe(dataframe, scaler):
    """
    Rescales a dataframe using given scaler.

    :param dataframe:   The dataframe to rescale
    :param scaler:      The scaler used for scaling of dataframe
    :return:            The rescaled dataframe
    """
    x = dataframe.values
    rescaled = scaler.inverse_transform(x)
    rescaled_dataframe = pd.DataFrame(rescaled, columns=dataframe.columns, index=dataframe.index)
    return rescaled_dataframe


def scale_inputs(x, y):
    """
    Downscales a dataframe x and a series y

    :param x:   The dataframe to downscale
    :param y:   The series to downscale
    :return:    The downscaled dataframe, series and their used scalers for rescaling later on
    """
    scaled_x, x_scaler = scale_dataframe(x)
    scaled_y, y_scaler = scale_series(y)
    return scaled_x, scaled_y, x_scaler, y_scaler


def rescale_input(x, y, x_scaler, y_scaler):
    """
    Rescales a scaled dataframe x and series y using their specified scalers used for downscaling.

    :param x:           The dataframe to rescale
    :param y:           The series to rescale
    :param x_scaler:    The scaler used for the dataframe
    :param y_scaler:    The scaler used for the series
    :return:            The rescaled dataframe and series
    """
    rescaled_x = rescale_dataframe(x, x_scaler)
    rescaled_y = rescale_series(y, y_scaler)
    return rescaled_x, rescaled_y


def estimate_knn(x, y):
    """
    Estimates the best parameters for k Nearest Neighbors given input samples and targets.
    Estimated parameters are: n_neighbors and weights.

    :param x:   The samples arraylike
    :param y:   The targets arraylike
    """
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
    """
    Estimates the best parameters for Decision Tree given input samples and targets.
    Estimated parameters are: max_depth.

    :param x:   The samples arraylike
    :param y:   The targets arraylike
    """
    decision_tree = GridSearchCV(tree.DecisionTreeRegressor(),
                                 param_grid={
                                     'max_depth': range(3, 25 + 1, 2)
                                 },
                                 n_jobs=-1)
    decision_tree.fit(x, y)
    print(decision_tree.best_params_, '\n', decision_tree.best_score_)


def estimate_random_forest(x, y):
    """
    Estimates the best parameters for Random Forest given input samples and targets.
    Estimated parameters are: n_estimators and max_depth.

    :param x:   The samples arraylike
    :param y:   The targets arraylike
    """
    random_forest = RandomizedSearchCV(ensemble.RandomForestRegressor(),
                                       param_distributions={
                                           'n_estimators': range(5, 125 + 1, 5),
                                           'max_depth': [None, 5, 10, 20],
                                       },
                                       n_iter=20,
                                       n_jobs=-1)
    random_forest.fit(x, y)
    print(random_forest.best_params_, '\n', random_forest.best_score_)


def estimate_linear_regression(x, y):
    """
    Estimates the best parameters for Linear Regression given input samples and targets.
    Estimated parameters are: normalize.

    :param x:   The samples arraylike
    :param y:   The targets arraylike
    """
    linear_regression = GridSearchCV(linear_model.LinearRegression(),
                                     param_grid={
                                         'normalize': [True, False]
                                     })
    linear_regression.fit(x, y)
    print(linear_regression.best_params_, '\n', linear_regression.best_score_)


def estimate_gru(x, y, rate):
    """
    Estimates best parameters for GRU given input samples and targets.
    Estimated parameters are: weights, dropout_rate, epochs, batch_size and learning rate.

    :param x:       The samples dataframe
    :param y:       The targets series
    :param rate:    The samplingrate ('D' or 'H')
    """
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


def estimate_arima(y, rate):
    start = time.time()
    model = auto_arima(y, start_p=1, start_q=1, max_p=4, max_q=4, error_action='ignore',
                       suppress_warnings=True, stepwise=True)
    runtime = time.time() - start
    print(model.summary(), '\n', runtime)


def estimate_ets(y, rate):
    start = time.time()
    model = ExponentialSmoothing(y)  # , trend=trendtype, freq=rate, damped=damped,
    #   seasonal=seasontype, seasonal_periods=seasonlength).fit(use_boxcox=box_cox)
    # TODO run GridSearch
    runtime = time.time() - start
    print('\n', runtime)  # TODO find best params


def estimate_prophet(x, y, rate):
    start = time.time()
    model = Prophet().fit(pd.DataFrame(data={
        'ds': x,
        'y': y
    }))
    runtime = time.time() - start
    print('\n', runtime)


def direct_parameter_estimation(x, y, rate):
    """
    Runs parameter estimation for all machine learning models for the given input

    :param x:       The samples to use
    :param y:       The targets to use
    :param rate:    The samplingrate ('D' or 'H')
    """
    # estimate_knn(x, y)
    # estimate_decision_tree(x, y)
    # estimate_random_forest(x, y)
    # estimate_linear_regression(x, y)
    estimate_gru(x, y, rate)


def timebased_parameter_estimation(x, y, rate):
    """
    Runs parameter estimation for all statistical models for the given input

    :param x:       The samples to use
    :param y:       The targets to use
    :param rate:    The samplingrate ('D' or 'H')
    """
    # estimate_ets(x, y, rate)
    # estimate_arima(x, y, rate)
    # estimate_prophet(x, y, rate)
    pass

def rotate_series(series):
    """
    Rotates a series to the left by one.
    The original first item becomes the last.

    :param series:  The series to rotate
    :return:        The rotated series
    """
    return pd.concat([series[1:], pd.Series(series.iloc[0])])


def model_testing(dataframe, pollutant, rate):
    """
    Tests all models on prediction using artificial features and also using the other pollutants directly

    :param dataframe:   The dataframe to use
    :param pollutant:   Which pollutant to predict
    :param rate:        The rate of sampling ('D' or 'H')
    """
    distance = 7 if rate == 'D' else 24  # distance for predictions, always 1 season (24 hrs or 7 days)

    series = dataframe[pollutant]
    rest = dataframe.drop(columns=[pollutant])[distance:]
    artificial = create_artificial_features(series, rate, steps=distance)[distance:]

    rotated = rotate_series(series[distance:])[:-1]
    timebased_parameter_estimation(artificial[:-1], rotated, rate)
    if len(rest.columns.tolist() > 1):
        timebased_parameter_estimation(rest[:-1], rotated, rate)
    
    for i in range(1, distance + 1):
        direct_parameter_estimation(artificial[:-i], rotated, rate)
        if len(rest.columns.tolist()) > 1:
            direct_parameter_estimation(rest[:-i], rotated, rate)

        rotated = rotate_series(rotated)[:-1]


def difference_series(series, stepsize=1):
    """
    Differentiates a Pandas Series object.
    The returned Series does not contain the first index of the input, because there is now sensible value for it.

    :param series:      The series to differentiate
    :param stepsize:    The stepsize to use for differentiation
    :return:            The differentiated Series
    """
    return series.diff(periods=stepsize).iloc[stepsize:]


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
    summed = start_level
    for i in range(0, len(series), stepsize):
        summed += series.iloc[i]
        dediff.append(summed)
    out = pd.Series(dediff)
    out.index = series.index
    return out


def difference_dataframe(dataframe, stepsize=1):
    """
    Differentiates all columns in a dataframe.

    :param dataframe:   The dataframe to differentiate
    :param stepsize:    The stepsize to use for differentiation
    :return:            The differentiated dataframe
    """
    return dataframe.diff(periods=stepsize).iloc[stepsize:]


def dediffference_dataframe(dataframe, start_row, stepsize=1):
    """
    Inverses the differentiation of a dataframe using a given start_row.

    :param dataframe:   The dataframe to dedifferentiate
    :param start_row:   The starting row values to inverse differentiation from
    :param stepsize:    The stepsize used for differentiation
    :return:            The dedifferentiated dataframe
    """
    dediff = pd.DataFrame()
    sums = start_row
    for i in range(0, len(dataframe), stepsize):
        sums = sums.add(dataframe.iloc[i])
        dediff = dediff.append(pd.DataFrame([sums]))
    dediff.index = dataframe.index
    return dediff


def test_pollutants(dataframe, rate):
    """
    Tests everything for all pollutants in a dataframe

    :param dataframe:   The dataframe to test
    :param rate:        The rate of the samples ('H' or 'D')
    """
    for pollutant in dataframe.columns:
        model_testing(dataframe, pollutant, rate)


if __name__ == '__main__':
    datadir = './post'
    modeldir = './models'
    statsfile = './stats.csv'
    files = glob.glob(datadir + '/*')

    for csv in files:
        station, steprate = get_info(csv, datadir)
        df = pd.read_csv(csv, index_col=0, parse_dates=[0], infer_datetime_format=True).drop(
            columns=['AirQualityStationEoICode', 'AveragingTime'])
        df = resample_dataframe(df, steprate)

        if len(df > 8760):
            df = df[:8760]

        test_pollutants(df, steprate)

        # stats_gru = RandomizedSearchCV()
        # stats_lstm = RandomizedSearchCV()
