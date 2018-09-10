import glob
import re
import sys
import time

import numpy as np
import pandas as pd
from fbprophet import Prophet
from pyramid.arima import ARIMA
from sklearn import neighbors, ensemble, tree, linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, median_absolute_error, \
    r2_score, explained_variance_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from main import create_gru, resample_dataframe, difference_dataframe, create_artificial_features, rotate_series, \
    scale_inputs, rescale_array


def deconstruct_save_string(savestring, folder):
    """
    Creates an information dict from the savestring

    :param savestring:  The save location
    :param folder:      The folder in which the file is saved
    :return:            The information dict
    """
    split = savestring.replace('.csv', '').replace(f'{folder}/', '').split('-')
    return {'station': split[0],
            'rate': split[1],
            'pollutant': split[2],
            'distance': split[3].replace('distance=', ''),
            'differenced': split[4].replace('differenced=', ''),
            'direct': split[5].replace('direct=', ''),
            'artificial': split[6].replace('artificial=', '') if len(split) == 7 else 'False'}


def get_arima_params(row):
    """
    Returns the first params for the arima model matching the given selector in row of dataframe

    :param row:          The dataframe to search in
    :return:            The parameters as dict
    """
    params = ['callback', 'disp', 'maxiter', 'method', 'order', 'scoring', 'scoring_args', 'seasonal_order', 'solver',
              'start_params', 'suppress_warnings', 'transparams', 'trend']
    params_dict = dict()
    for k in params:
        if k == 'order' or k == 'seasonal_order' or k == 'scoring_args':
            value = eval(row[(row.model == 'arima')][k])
        else:
            value = row[(row.model == 'arima')][k]

        if not (type(value) is np.float64 and np.isnan(value)):
            if type(value) is np.float64:
                value = int(value)
            params_dict[k] = value
    return params_dict


def get_ets_params(row):
    """
    Returns the first params for the ets model matching the given selector in row of dataframe

    :param row:          The dataframe to search in
    :return:            The parameters as dict
    """
    params = ['smoothing_level', 'smoothing_slope', 'smoothing_seasonal', 'damping_slope', 'use_boxcox',
              'initial_level', 'initial_slope', 'initial_seasons', 'lamda', 'remove_bias']
    params_dict = dict()
    for k in params:
        if k == 'initial_seasons':
            value = eval(re.sub(' +(\n)*', ', ', row[(row.model == 'ets')][k]))
        else:
            value = row[(row.model == 'ets')][k]
        params_dict[k] = value
    return params_dict


def get_knn_params(row):
    """
    Returns the first params for the knn model matching the given selector in row of dataframe

    :param row:          The dataframe to search in
    :return:            The parameters as dict
    """
    params = ['n_neighbors', 'weights']
    params_dict = dict()
    for k in params:
        value = row[(row.model == 'knn')][k]
        if type(value) is np.float64:
            value = int(value)
        params_dict[k] = value
    return params_dict


def get_decision_tree_params(row):
    """
    Returns the first params for the decision tree model matching the given selector in row of dataframe

    :param row:          The dataframe to search in
    :return:            The parameters as dict
    """
    params = ['max_depth']
    params_dict = dict()
    for k in params:
        value = row[(row.model == 'decision_tree')][k]
        if type(value) is np.float64:
            value = int(value)
        params_dict[k] = value
    return params_dict


def get_random_forest_params(row):
    """
    Returns the first params for the random forest model matching the given selector in row of dataframe

    :param row:          The dataframe to search in
    :return:            The parameters as dict
    """
    params = ['n_estimators', 'max_depth']
    params_dict = dict()
    for k in params:
        value = row[(row.model == 'random_forest')][k]
        if type(value) is np.float64:
            value = int(value)
        params_dict[k] = value
    return params_dict


def get_linear_regression_params(row):
    """
    Returns the first params for the linear regression model matching the given selector in row of dataframe

    :param row:          The dataframe to search in
    :return:            The parameters as dict
    """
    params = ['normalize']
    params_dict = dict()
    for k in params:
        value = row[(row.model == 'linear_regression')][k]
        params_dict[k] = value
    return params_dict


def get_gru_params(row):
    """
    Returns the first params for the gru model matching the given selector in row of dataframe

    :param row:         The dataframe to search in
    :return:            The parameters as dict
    """
    params = ['weights', 'dropout_rate', 'input_shape', 'epochs', 'batch_size', 'learning_rate']
    params_dict = dict()
    for k in params:
        value = row[(row.model == 'gru')][k]
        if k == 'weights' or k == 'epochs' or k == 'batch_size':
            value = int(value)
        elif k == 'input_shape':
            value = eval(value)
        params_dict[k] = value
    return params_dict


def score_prediction(actual, predicted):
    return {'mean_squared_error': mean_squared_error(actual, predicted),
            'mean_squared_log_error': mean_squared_log_error(actual, predicted),
            'mean_absolute_error': mean_absolute_error(actual, predicted),
            'median_absolute_error': median_absolute_error(actual, predicted),
            'r2': r2_score(actual, predicted),
            'explained_variance': explained_variance_score(actual, predicted)}


def evaluate_knn(row, x, y, distance):
    """
    Evaluate the model with the params given in the row

    :param row:         The row to get the parameters from
    :param x:           The x input vector
    :param y:           The y target vector
    :param distance:    The distance to predict for
    :return:
    """
    used_x = x[:-distance]
    used_y = y[:-distance]
    params = get_knn_params(row)
    times = dict()

    start = time.clock()
    model = neighbors.KNeighborsRegressor(**params)
    fit = model.fit(used_x, used_y)
    times['fit_time'] = time.clock() - start

    start = time.clock()
    prediction = fit.predict(x[-distance:])
    times['prediction_time'] = time.clock() - start

    scores = score_prediction(y[-distance:], prediction)

    return {**times, **scores}


def evaluate_decision_tree(row, x, y, distance):
    """
    Evaluate the model with the params given in the row

    :param row:         The row to get the parameters from
    :param x:           The x input vector
    :param y:           The y target vector
    :param distance:    The distance to predict for
    :return:
    """
    used_x = x[:-distance]
    used_y = y[:-distance]
    params = get_decision_tree_params(row)
    times = dict()

    start = time.clock()
    model = tree.DecisionTreeRegressor(**params)
    fit = model.fit(used_x, used_y)
    times['fit_time'] = time.clock() - start

    start = time.clock()
    prediction = fit.predict(x[-distance:])
    times['prediction_time'] = time.clock() - start

    scores = score_prediction(y[-distance:], prediction)

    return {**times, **scores}


def evaluate_random_forest(row, x, y, distance):
    """
    Evaluate the model with the params given in the row

    :param row:         The row to get the parameters from
    :param x:           The x input vector
    :param y:           The y target vector
    :param distance:    The distance to predict for
    :return:
    """
    used_x = x[:-distance]
    used_y = y[:-distance]
    params = get_random_forest_params(row)
    times = dict()

    start = time.clock()
    model = ensemble.RandomForestRegressor(**params)
    fit = model.fit(used_x, used_y)
    times['fit_time'] = time.clock() - start

    start = time.clock()
    prediction = fit.predict(x[-distance:])
    times['prediction_time'] = time.clock() - start

    scores = score_prediction(y[-distance:], prediction)

    return {**times, **scores}


def evaluate_linear_regression(row, x, y, distance):
    """
    Evaluate the model with the params given in the row

    :param row:         The row to get the parameters from
    :param x:           The x input vector
    :param y:           The y target vector
    :param distance:    The distance to predict for
    :return:
    """
    used_x = x[:-distance]
    used_y = y[:-distance]
    params = get_linear_regression_params(row)
    times = dict()

    start = time.clock()
    model = linear_model.LinearRegression(**params)
    fit = model.fit(used_x, used_y)
    times['fit_time'] = time.clock() - start

    start = time.clock()
    prediction = fit.predict(x[-distance:])
    times['prediction_time'] = time.clock() - start

    scores = score_prediction(y[-distance:], prediction)

    return {**times, **scores}


def evaluate_gru(row, x, y, distance):
    """
    Evaluate the model with the params given in the row

    :param row:         The row to get the parameters from
    :param x:           The x input vector
    :param y:           The y target vector
    :param distance:    The distance to predict for
    :return:
    """
    from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

    scaled_x, scaled_y, x_scaler, y_scaler = scale_inputs(x, y)
    scaled_x = scaled_x.values.reshape(scaled_x.shape[0], scaled_x.shape[1], 1)
    scaled_y = scaled_y.values
    used_x = scaled_x[:-distance]
    used_y = scaled_y[:-distance]
    params = get_gru_params(row)
    times = dict()

    start = time.clock()
    model = KerasRegressor(build_fn=create_gru, **params)
    fit = model.fit(used_x, used_y)
    times['fit_time'] = time.clock() - start

    start = time.clock()
    prediction = rescale_array(model.predict(scaled_x[-distance:]), y_scaler)
    times['prediction_time'] = time.clock() - start

    scores = score_prediction(y[-distance:], prediction)

    return {**times, **scores}


def evaluate_ets(row, y, distance, rate):
    """
    Evaluate the model with the params given in the row

    :param row:         The row to get the parameters from
    :param y:           The y target vector
    :param distance:    The distance to predict for
    :param rate:        The sampling rate for the values in y
    :return:
    """
    used_y = y[:-distance]
    periods = int(len(used_y) / rate)
    params = get_ets_params(row)
    times = dict()

    start = time.clock()
    model = ExponentialSmoothing(used_y, trend='add', seasonal_periods=periods, seasonal='add')
    model.params = params
    fit = model.fit()
    times['fit_time'] = time.clock() - start

    start = time.clock()
    prediction = fit.forecast(distance)
    times['prediction_time'] = time.clock() - start

    scores = score_prediction(y[-distance:], prediction)

    return {**times, **scores}


def evaluate_arima(row, y, distance):
    """
    Evaluate the model with the params given in the row

    :param row:         The row to get the parameters from
    :param y:           The y target vector
    :param distance:    The distance to predict for
    :return:
    """
    used_y = y[:-distance]
    params = get_arima_params(row)
    times = dict()

    start = time.clock()
    model = ARIMA(**params)
    fit = model.fit(y=used_y)
    times['fit_time'] = time.clock() - start

    start = time.clock()
    prediction = fit.predict(distance)
    times['prediction_time'] = time.clock() - start

    scores = score_prediction(y[-distance:], prediction)

    return {**times, **scores}


def evaluate_prophet(y, distance, rate):
    """
    Evaluate the model with the params given in the row

    :param y:           The y target vector
    :param distance:    The distance to predict for
    :param rate:        The sampling rate for the values in y
    :return:
    """
    used_y = y[:-distance]
    times = dict()

    start = time.clock()
    prophet = Prophet().fit(pd.DataFrame(data={
        'ds': used_y.index,
        'y': used_y
    }))
    times['fit_time'] = time.clock() - start

    start = time.clock()
    future = prophet.make_future_dataframe(distance, rate)
    complete_prediction = prophet.predict(future)
    prediction = complete_prediction['yhat'][-distance:]
    times['prediction_time'] = time.clock() - start

    scores = score_prediction(y[-distance:], prediction)

    return {**times, **scores}


def evaluate_best_params(resources, results_folder, debugging=False):
    for csv in glob.glob(f'{resources}/*.csv'):
        station, rate = csv.replace(f'{resources}/', '').replace('.csv', '').replace('day', 'D').replace('hour',
                                                                                                         'H').split('-')
        results = glob.glob(f'{results_folder}/{station}-{rate}*.csv')
        data_df = pd.read_csv(csv, index_col=0, parse_dates=[0], infer_datetime_format=True).drop(
            columns=['AirQualityStationEoICode', 'AveragingTime'])

        if len(data_df > 8760):
            data_df = data_df[:8760]

        if debugging:
            data_df = data_df[:debug_len]

        data_df = resample_dataframe(data_df, rate)
        differenced_df = difference_dataframe(data_df)

        stats_df = pd.DataFrame()
        for result in results:
            info = deconstruct_save_string(result, results_folder)
            current_df = pd.read_csv(result, index_col=0)

            for k, v in info.items():
                current_df[k] = v

            stats_df = pd.concat([stats_df, current_df])

        stats_df = stats_df.reset_index().drop(columns=['level_0'])
        best_stats_df = pd.DataFrame()
        # TODO: save stats accordingly to df

        for idx, r in stats_df.iterrows():
            pollutant, distance, differenced, direct, artificial, model = r['pollutant'], int(r['distance']), r[
                'differenced'], r['direct'], r['artificial'], r['model']

            if differenced:
                used_df = differenced_df
            else:
                used_df = data_df
            series = used_df[pollutant]

            if direct:
                rotated = series
                for i in range(0, distance):
                    rotated = rotate_series(rotated)[:-1]
                rest = used_df.drop(columns=[pollutant])

                if artificial:
                    steps = 7 if rate == 'D' else 24
                    x = create_artificial_features(series, rate, steps)[steps:]
                    y = rotated[steps:]
                else:
                    x = rest
                    y = rotated

                if model == 'knn':
                    evaluate_knn(r, x, y, distance)
                elif model == 'decision_tree':
                    evaluate_decision_tree(r, x, y, distance)
                elif model == 'random_forest':
                    evaluate_random_forest(r, x, y, distance)
                elif model == 'linear_regression':
                    evaluate_linear_regression(r, x, y, distance)
                elif model == 'gru':
                    evaluate_gru(r, x, y, distance)
            else:
                if model == 'arima':
                    evaluate_arima(r, series, distance)
                elif model == 'ets':
                    evaluate_ets(r, series, distance, rate)
                elif model == 'prophet':
                    evaluate_prophet(series, distance, rate)


if __name__ == '__main__':
    debug_len = 200
    debug = not sys.gettrace() is None

    resource_loc = 'post'
    results_loc = 'results'
    models = ['knn', 'decision_tree', 'random_forest', 'linear_regression', 'gru', 'ets', 'arima', 'prophet']
    evaluate_best_params(resource_loc, results_loc, debug)
