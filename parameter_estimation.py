import calendar
import glob
import os
import sys
import time



# multiprocessing.set_start_method('forkserver')

import numpy as np
import pandas as pd
from fbprophet import Prophet
from joblib import Parallel, delayed
from pyramid.arima import auto_arima
from sklearn import neighbors, ensemble, tree, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from logger import Logger


def create_artificial_features(series, frequency='H', steps=7, weekdays=False, months=False, statistical=True):
    """
    Creates artificial features for a given series with Timestamp Index

    :param series:      the base series to use
    :param frequency:   the frequency of values in the series
    :param steps:       the amount of steps to lag the series by
    :param weekdays:    add one hot encoding for weekdays
    :param months:      add one hot encoding for months
    :return:            the dataframe containing the artificial features for the input series
    """
    # interpolated = series.interpolate(method='time', frequency=frequency)
    lagged = create_lagged_features(series, frequency, steps)
    statistics = lagged

    if statistical:
        statistics['sum'] = lagged.sum(axis=1)
        statistics['mean'] = lagged.mean(axis=1)
        statistics['median'] = lagged.median(axis=1)

    if weekdays:
        weekdays_df = pd.get_dummies(lagged.index.weekday_name)
        weekdays_df = weekdays_df.applymap(lambda x: bool(x))
        weekdays_df.index = lagged.index
        statistics = statistics.join(weekdays_df)

    if months:
        months_df = pd.get_dummies(lagged.index.month.map(lambda x: calendar.month_abbr[x]))
        months_df = months_df.applymap(lambda x: bool(x))
        months_df.index = lagged.index
        statistics = statistics.join(months_df)

    return statistics


def create_lagged_features(series, frequency='H', steps=7):
    """
    Creates a dataframe from a series containing the original series and the lagged values for the specified amount of
    steps.

    :param series:      the series to use
    :param frequency:   the frequency of the values by which to shift
    :param steps:       the amount of steps to shift
    :return:            the shifted dataframe
    """
    lagged = pd.DataFrame()

    for i in range(1, steps + 1):
        lagged['lag {}{}'.format(i, frequency)] = series.shift(i, freq=frequency)

    lagged.index = series.index

    return lagged.interpolate()


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
    return dataframe.dropna()


def get_info(csv_path, directory):
    """
    Gets station name and rate from a csv path string

    :param csv_path:    The path string
    :param directory:   The directory of the csv
    :return:            The station name and rate
    """
    info = csv_path.replace(f'{directory}/', '').replace('.csv', '').split('-')
    station = info[0]
    rate = {'day': 'D', 'hour': 'H'}[info[-1]]
    return station, rate


def create_gru(weights, input_shape, dropout_rate, learning_rate):
    """
    Creates a GRU based RNN using the given input parameters.

    :param weights:         The amount of output weights for GRU layer
    :param input_shape:     The shape of the inputs
    :param dropout_rate:    The dropout rate after GRU layer
    :param learning_rate:   The learning rate of the optimizer
    :return:                The compiled model
    """
    from tensorflow.python.keras import Sequential, optimizers
    from tensorflow.python.keras.layers import GRU, Dropout, Dense

    optimizer = optimizers.RMSprop(lr=learning_rate)

    model = Sequential()
    model.add(GRU(weights, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer, loss='mse')
    return model


def scale_array(array, feature_range=(-1, 1)):
    x = array
    x = x.reshape(len(x), 1)
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler = scaler.fit(x)
    scaled = scaler.transform(x)
    scaled = scaled.reshape(len(scaled))
    return scaled, scaler


def scale_series(series, feature_range=(-1, 1)):
    """
    Downscales a series to given feature_range.

    :param series:          The series to downscale
    :param feature_range:   The range of values after scaling
    :return:                The scaled series and used scaler
    """
    x = series.values
    scaled, scaler = scale_array(x, feature_range)
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


def rescale_array(array, scaler):
    x = array
    x = x.reshape(len(x), 1)
    rescaled = scaler.inverse_transform(x)
    rescaled = rescaled.reshape(len(rescaled))
    return rescaled


def rescale_series(series, scaler):
    """
    Rescales a series using given scaler.

    :param series:  The series to rescale
    :param scaler:  The scaler used to scale the series
    :return:        The rescaled series
    """
    x = series.values
    rescaled = rescale_array(x, scaler)
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
    logger.log('Finding best KNN parameters', 3)
    knn = RandomizedSearchCV(neighbors.KNeighborsRegressor(),
                             param_distributions={
                                 'n_neighbors': range(2, 50 + 1, 2),
                                 'weights': ['uniform', 'distance']
                             },
                             cv=TimeSeriesSplit(),
                             scoring='neg_mean_squared_error',
                             verbose=2,
                             n_iter=20,
                             n_jobs=-1)
    knn.fit(x, y)
    logger.log(f'Found best KNN model with params {knn.best_params_} and score {knn.best_score_}', 3)

    return knn.best_params_, knn.best_score_


def estimate_decision_tree(x, y):
    """
    Estimates the best parameters for Decision Tree given input samples and targets.
    Estimated parameters are: max_depth.

    :param x:   The samples arraylike
    :param y:   The targets arraylike
    """
    logger.log('Finding best Decision Tree parameters', 3)
    decision_tree = GridSearchCV(tree.DecisionTreeRegressor(),
                                 param_grid={
                                     'max_depth': range(3, 25 + 1, 2)
                                 },
                                 cv=TimeSeriesSplit(),
                                 scoring='neg_mean_squared_error',
                                 verbose=2,
                                 n_jobs=-1)
    decision_tree.fit(x, y)
    logger.log(f'Found best Decision Tree model with params {decision_tree.best_params_}'
               f' and score {decision_tree.best_score_}', 3)

    return decision_tree.best_params_, decision_tree.best_score_


def estimate_random_forest(x, y):
    """
    Estimates the best parameters for Random Forest given input samples and targets.
    Estimated parameters are: n_estimators and max_depth.

    :param x:   The samples arraylike
    :param y:   The targets arraylike
    """
    logger.log(f'Finding best Random Forest parameters', 3)
    random_forest = RandomizedSearchCV(ensemble.RandomForestRegressor(),
                                       param_distributions={
                                           'n_estimators': range(5, 125 + 1, 5),
                                           'max_depth': [None, 5, 10, 20],
                                       },
                                       cv=TimeSeriesSplit(),
                                       scoring='neg_mean_squared_error',
                                       verbose=2,
                                       n_iter=20,
                                       n_jobs=-1)
    random_forest.fit(x, y)
    logger.log(f'Found best Random Forest model with params {random_forest.best_params_}'
               f' and score {random_forest.best_score_}', 3)

    return random_forest.best_params_, random_forest.best_score_


def estimate_linear_regression(x, y):
    """
    Estimates the best parameters for Linear Regression given input samples and targets.
    Estimated parameters are: normalize.

    :param x:   The samples arraylike
    :param y:   The targets arraylike
    """
    logger.log('Finding best Linear Regression parameters', 3)
    linear_regression = GridSearchCV(linear_model.LinearRegression(),
                                     param_grid={
                                         'normalize': [True, False]
                                     },
                                     cv=TimeSeriesSplit(),
                                     scoring='neg_mean_squared_error',
                                     verbose=2,
                                     n_jobs=-1)
    linear_regression.fit(x, y)
    logger.log(f'Found best Linear Regression model with params {linear_regression.best_params_}'
               f' and score {linear_regression.best_score_}', 3)

    return linear_regression.best_params_, linear_regression.best_score_


def tensorflow_score(estimator, x, y, scaler, batch_size, **kwargs):
    """
    Calculates the negative mse for a Keras model by rescaling the predictions and truths.

    :param estimator:   The estimator
    :param x:           The input values
    :param y:           The base truth
    :param batch_size:  The batch size used as input length for training
    :param scaler:      The scaler to use for rescaling
    :batch_size:        The batch size to use for inputs
    :param kwargs:      kwargs
    :return:            The inverted loss function
    """
    from keras.models import Sequential
    kwargs = estimator.filter_sk_params(Sequential.evaluate, kwargs)
    prediction = estimator.model.predict(x, batch_size, kwargs)
    rescaled_prediction = rescale_array(prediction, scaler)
    rescaled_y = rescale_array(y, scaler)
    mse = mean_squared_error(rescaled_y, rescaled_prediction)
    return -mse


def estimate_gru(x, y, batch_size):
    """
    Estimates best parameters for GRU given input samples and targets.
    Estimated parameters are: weights, dropout_rate, epochs, batch_size and learning rate.

    :param x:           The samples dataframe
    :param y:           The targets series
    :param batch_size:  The batch size for the gru
    """
    from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
    from tensorflow.python.keras.backend import clear_session
    clear_session()
    logger.log('Finding best GRU parameters', 3)
    x, y, x_scaler, y_scaler = scale_inputs(x, y)
    x = x.values.reshape(x.shape[0], x.shape[1], 1)
    y = y.values
    # batch_size = [24, 24 * 7] if rate == 'H' else [7, 7 * 30]

    gru = RandomizedSearchCV(KerasRegressor(create_gru, verbose=0),
                             param_distributions={
                                 'weights': np.linspace(1, 100, 20, endpoint=True, dtype=int),
                                 'dropout_rate': np.linspace(0.1, 0.3, 3, endpoint=True),
                                 'input_shape': [(x.shape[1], x.shape[2])],
                                 'epochs': range(1, 10 + 1),
                                 'batch_size': [batch_size],
                                 'learning_rate': np.linspace(0.001, 0.02, 10, endpoint=True)
                             },
                             cv=TimeSeriesSplit(),
                             scoring=lambda estimator, X, y, **kwargs: tensorflow_score(estimator, X, y, y_scaler,
                                                                                        batch_size, **kwargs),
                             verbose=2,
                             n_jobs=-1,
                             n_iter=20)
    gru.fit(x, y)
    logger.log(f'Found best GRU model with params {gru.best_params_} and score {gru.best_score_}', 3)

    return gru.best_params_, gru.best_score_


def estimate_arima(y, distance):
    """
    Estimates best parameters for ARIMA model and input series y.

    :param y:           The series to predict further
    :param distance:    The amount of steps to predict
    """
    logger.log('Finding best ARIMA parameters', 3)
    start = time.time()
    model = auto_arima(y, start_p=1, start_q=1, max_p=4, max_q=4, error_action='ignore',
                       suppress_warnings=True, stepwise=False, n_jobs=-1, out_of_sample_size=distance, scoring='mse')
    prediction = model.predict(distance)
    mse = mean_squared_error(y[-distance:], prediction)
    runtime = time.time() - start
    logger.log(f'Found best ARIMA model with mse {mse} in {runtime}', 3)

    return model.get_params(), mse


def estimate_ets(y, distance, rate):
    """
    Estimates the best parameters for ETS prediction of series y

    :param y:           The series to predict
    :param distance:    The amount of steps to predict
    :param rate:        The sampling rate used (D/H)
    :return:
    """

    def get_ets_stats(trend, season, damped, box_cox):
        print(f'running ets trend={trend}, damped={damped}, season={season}, box_cox={box_cox}')
        model = ExponentialSmoothing(fitting, trend=trend, seasonal=season, damped=damped,
                                     seasonal_periods=seasonal_periods)
        fit = model.fit(use_boxcox=box_cox)
        prediction = fit.forecast(distance)
        prediction = prediction[~np.isnan(prediction)]
        mse = mean_squared_error(y[-len(prediction):], prediction)

        return fit.params, mse

    if rate == 'H':
        fitting = y[:-(distance * 7)]
        seasonal_periods = int(len(fitting) / (24 * 7))
    else:
        fitting = y[:-distance]
        seasonal_periods = int(len(fitting) / 7)

    logger.log('Finding best ETS model', 3)
    start = time.time()

    # add_mul = ['additive', 'multiplicative']
    t_f = [True, False]
    has_negatives = y.min() <= 0

    matrix = list()
    trend = 'additive'  # because of nan errors otherwise
    for season in ['additive']:  # because of differencing problems otherwise
        for damped in t_f:
            for box_cox in t_f:
                # only use box_cox if no negative values in input
                if not (has_negatives and box_cox is True):
                    matrix.append((season, damped, box_cox))

    stats = Parallel(n_jobs=-1)(delayed(get_ets_stats)(trend, params[0], params[1], params[2]) for params in matrix)

    best_mse = stats[0][1]
    best_params = stats[0][0]

    runtime = time.time() - start
    logger.log(f'Found best ETS model with mse {best_mse} and params {best_params} in {runtime}', 3)

    return best_params, best_mse


def estimate_prophet(y, distance, rate):
    """
    Estimate the best parameters for prophet prediction

    :param y:           The series to predict for
    :param distance:    The amount of steps to predict
    :param rate:        The rate of sampling ('D' or 'H')
    """
    logger.log(f'Finding best Prophet model', 3)
    start = time.time()
    model = Prophet().fit(pd.DataFrame(data={
        'ds': y.index[:-distance],
        'y': y[:-distance]
    }))
    mse = mean_squared_error(y[-distance:],
                             model.predict(model.make_future_dataframe(distance, rate))['yhat'][-distance:])
    runtime = time.time() - start
    logger.log(f'Found best Prophet model with mse {mse} in {runtime}', 3)

    return mse


def direct_parameter_estimation(x, y):
    """
    Runs parameter estimation for all machine learning models for the given input

    :param x:       The samples to use
    :param y:       The targets to use
    """
    params = dict()
    scores = dict()

    params['knn'], scores['knn'] = estimate_knn(x, y)
    params['decision_tree'], scores['decision_tree'] = estimate_decision_tree(x, y)
    params['random_forest'], scores['random_forest'] = estimate_random_forest(x, y)
    params['linear_regression'], scores['linear_regression'] = estimate_linear_regression(x, y)
    params['gru'], scores['gru'] = estimate_gru(x, y, len(x.columns))

    return params, scores


def timebased_parameter_estimation(y, distance, rate):
    """
    Runs parameter estimation for all statistical models for the given input

    :param y:           The targets to use
    :param distance:    The amount of timesteps to predict
    :param rate:        The samplingrate ('D' or 'H')
    """
    params = dict()
    scores = dict()

    params['ets'], scores['ets'] = estimate_ets(y, distance, rate)
    try:
        params['arima'], scores['arima'] = estimate_arima(y, distance)
    except ValueError:
        logger.log(
            'Skipping ARIMA: Could not successfully fit ARIMA to input data. It is likely your data is non-stationary.',
            3)
    scores['prophet'] = estimate_prophet(y, distance, rate)
    params['prophet'] = dict()

    return params, scores


def rotate_series(series):
    """
    Rotates a series to the left by one.
    The original first item becomes the last.

    :param series:  The series to rotate
    :return:        The rotated series
    """
    return pd.concat([series[1:], pd.Series(series.iloc[0])])


def save_results(params, scores, savepath):
    """
    Save parameters and scores to specified path

    :param params:      The model parameters dict
    :param scores:      The model scores dict
    :param savepath:    The path to save the file at
    """
    if not os.path.isfile(savepath):
        open(savepath, 'x')

    dataframe = pd.DataFrame()
    for model, param in params.items():
        for k, v in param.items():
            if type(v) is tuple:
                param[k] = str(v)

        currentframe = pd.DataFrame().from_records([param.values()], columns=param.keys())
        currentframe['score'] = scores[model]
        currentframe['model'] = model
        dataframe = pd.concat([dataframe, currentframe])

    dataframe = dataframe.reset_index()
    dataframe.to_csv(savepath)


def build_save_string(station, pollutant, distance, differenced, direct, artificial, rate):
    """
    Builds the save location string for the given inputs.
    The filetype will be a csv file.

    :param station:     The station for which the data is saved
    :param pollutant:   The pollutant
    :param distance:    The forecast distance
    :param differenced: If the dataframe was differenced
    :param direct:      If the forecast uses a direct model
    :param artificial:  If the forecast used the artificial set or other pollutants
    :param rate:        The used sampling rate (D/H)
    :return:            The save location string
    """
    savepath = f'./results/{station}-{rate}-{pollutant}-distance={distance}-differenced={differenced}-direct={direct}'
    if direct:
        savepath += f'-artificial={artificial}'
    savepath += '.csv'
    return savepath


def model_testing(dataframe, pollutant, station, rate, differenced):
    """
    Tests all models on prediction using artificial features and also using the other pollutants directly

    :param dataframe:   The dataframe to use
    :param pollutant:   Which pollutant to predict
    :param station:     The station to test for
    :param rate:        The rate of sampling ('D' or 'H')
    :param differenced: If the dataframe was differenced
    """
    distance = 7 if rate == 'D' else 24  # distance for predictions, always 1 season (24 hrs or 7 days)
    use_values = [1, 2, 3, 4, 5, 6, 7] if distance == 7 else [1, 2, 3, 5, 10, 12, 24]

    series = dataframe[pollutant]
    rest = dataframe.drop(columns=[pollutant])[distance:]
    artificial = create_artificial_features(series, rate, steps=distance)[distance:]

    save_path = build_save_string(station, pollutant, distance, differenced, False, artificial, rate)
    if not os.path.exists(save_path):
        logger.log('Running tests for timebased models', 2)
        timebased_params, timebased_scores = timebased_parameter_estimation(series, distance, rate)
        save_results(timebased_params, timebased_scores, save_path)
    else:
        logger.log('Skipping tests for timebased models because they already exist', 2, False)

    rotated = series[distance:]
    for i in range(1, distance + 1):
        rotated = rotate_series(rotated)[:-1]

        # only use specific predefined forecast distances.
        if i in use_values:
            save_path = build_save_string(station, pollutant, i, differenced, True, True, rate)
            if not os.path.exists(save_path):
                logger.log(f'Running tests for direct models on artificial set with distance {i}{rate}', 2)
                artificial_params, artificial_scores = direct_parameter_estimation(artificial[:-i], rotated)
                save_results(artificial_params, artificial_scores, save_path)
            else:
                logger.log(f'Skipping tests for direct models on artificial set with distance {i}{rate},'
                           f' because they already exist', 2, False)

            if len(rest.columns.tolist()) > 1:
                save_path = build_save_string(station, pollutant, i, differenced, True, False, rate)
                if not os.path.exists(save_path):
                    logger.log(f'Running tests for direct models on direct set with distance {i}{rate}', 2)
                    direct_params, direct_scores = direct_parameter_estimation(rest[:-i], rotated)
                    save_results(direct_params, direct_scores, save_path)
                else:
                    logger.log(f'Skipping tests for direct models on direct set with distance {i}{rate},'
                               f' because they already exist', 2, False)


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


def test_pollutants(dataframe, station, rate):
    """
    Tests everything for all pollutants in a dataframe

    :param dataframe:   The dataframe to test
    :param station:     The station to test
    :param rate:        The rate of the samples ('H' or 'D')
    """
    for pollutant in dataframe.columns:
        logger.log(f'Running tests for {pollutant}', 1)
        model_testing(dataframe, pollutant, station, rate, False)

        logger.log(f'Running tests for differenced {pollutant}', 1)
        model_testing(difference_dataframe(dataframe), pollutant, station, rate, True)


def find_best_params(data_dir):
    """
    Find best params for all models using data from the given datadir

    :param data_dir:    The directory containing the converted eea csvs
    """
    files = glob.glob(data_dir + '/*.csv')
    file_num = 0
    for csv in files:
        file_num += 1
        logger.log(f'({file_num}/{len(files)})\t\tRunning tests for {csv}')
        station_name, steprate = get_info(csv, datadir)
        df = pd.read_csv(csv, index_col=0, parse_dates=[0], infer_datetime_format=True).drop(
            columns=['AirQualityStationEoICode', 'AveragingTime'])
        df = resample_dataframe(df, steprate)

        if len(df > 8760):
            df = df[:8760]

        if debug:
            df = df[:debug_len]

        test_pollutants(df, station_name, steprate)


if __name__ == '__main__':
    os.environ["LOKY_PICKLER"] = 'cloudpickle'
    logger = Logger('./event.log')
    datadir = './post'
    modeldir = './models'
    statsfile = './stats.csv'

    debug_len = 200
    debug = not sys.gettrace() is None
    if debug:
        logger.log(f'Running in debugger, dataframes will be cut to {debug_len} elements')

    if not os.path.exists('./results'):
        os.makedirs('./results')

    while True:
        try:
            find_best_params(datadir)
            sys.exit(0)
        except KeyboardInterrupt:
            sys.exit(1)
        except:
            logger.log('Main crashed, restarting')
