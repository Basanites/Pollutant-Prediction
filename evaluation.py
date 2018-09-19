import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from fbprophet import Prophet
from joblib import Parallel, delayed
from pyramid.arima import ARIMA
from sklearn import neighbors, ensemble, tree, linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score, \
    explained_variance_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from parameter_estimation import create_gru, resample_dataframe, difference_dataframe, create_artificial_features, \
    rotate_series, scale_inputs, rescale_array


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
            value = eval(row[k])
        else:
            value = row[k]

        if type(value) is np.float64 or type(value) is float:
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
            value = eval(re.sub(' +(\n)*', ', ', row[k]))
        else:
            value = row[k]
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
        value = row[k]
        if type(value) is np.float64 or type(value) is float:
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
        value = row[k]
        if type(value) is np.float64 or type(value) is float:
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
        value = row[k]
        if (type(value) is np.float64 or type(value) is float) and not np.isnan(value):
            value = int(value)
        if not np.isnan(value):
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
        value = row[k]
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
        value = row[k]
        if k == 'weights' or k == 'epochs' or k == 'batch_size':
            value = int(value)
        elif k == 'input_shape':
            value = eval(value)
        params_dict[k] = value
    return params_dict


def score_prediction(actual, predicted, norm_factor=None):
    """
    Returns all possible regression scores for the given timeseries'

    :param actual:      The expected timeseries values
    :param predicted:   The predicted timeseries values
    :param norm_factor: The factor to use for normalization of the scores
    :return:            The dict containing the scores
    """
    scores = {'mean_squared_error': mean_squared_error(actual, predicted),
              'mean_absolute_error': mean_absolute_error(actual, predicted),
              'median_absolute_error': median_absolute_error(actual, predicted),
              'r2': r2_score(actual, predicted),
              'explained_variance': explained_variance_score(actual, predicted)}

    if norm_factor:
        scores = {**scores, **{
            'norm_mean_squared_error': scores['mean_squared_error'] * norm_factor,
            'norm_mean_absolute_error': scores['mean_absolute_error'] * norm_factor,
            'norm_median_absolute_error': scores['median_absolute_error'] * norm_factor,
            'norm_r2': scores['r2'] * norm_factor,
            'norm_explained_variance': scores['explained_variance'] * norm_factor
        }}

    return scores


def evaluate_knn(row, x, y, validation_size):
    """
    Evaluate the model with the params given in the row

    :param row:         The row to get the parameters from
    :param x:           The x input vector
    :param y:           The y target vector
    :param validation_size:    The validation_size to predict for
    :return:
    """
    used_x = x[:-validation_size]
    used_y = y[:-validation_size]
    params = get_knn_params(row)
    times = dict()

    start = time.clock()
    model = neighbors.KNeighborsRegressor(**params)
    fit = model.fit(used_x, used_y)
    times['fit_time'] = time.clock() - start

    start = time.clock()
    prediction = fit.predict(x[-validation_size:])
    norm_factor = 1 / (prediction.max() - prediction.min())
    times['prediction_time'] = time.clock() - start

    scores = score_prediction(y[-validation_size:], prediction, norm_factor)

    return {'params': params, 'prediction': prediction, **times, **scores, 'norm_factor': norm_factor}


def evaluate_decision_tree(row, x, y, validation_size):
    """
    Evaluate the model with the params given in the row

    :param row:             The row to get the parameters from
    :param x:               The x input vector
    :param y:               The y target vector
    :param validation_size: The validation distance to use for scoring
    :return:
    """
    used_x = x[:-validation_size]
    used_y = y[:-validation_size]
    params = get_decision_tree_params(row)
    times = dict()

    start = time.clock()
    model = tree.DecisionTreeRegressor(**params)
    fit = model.fit(used_x, used_y)
    times['fit_time'] = time.clock() - start

    start = time.clock()
    prediction = fit.predict(x[-validation_size:])
    norm_factor = 1 / (prediction.max() - prediction.min())
    times['prediction_time'] = time.clock() - start

    scores = score_prediction(y[-validation_size:], prediction, norm_factor)

    return {'params': params, 'prediction': prediction, **times, **scores, 'norm_factor': norm_factor}


def evaluate_random_forest(row, x, y, validation_size):
    """
    Evaluate the model with the params given in the row

    :param row:             The row to get the parameters from
    :param x:               The x input vector
    :param y:               The y target vector
    :param validation_size: The validation distance to use for scoring
    :return:
    """
    used_x = x[:-validation_size]
    used_y = y[:-validation_size]
    params = get_random_forest_params(row)
    times = dict()

    start = time.clock()
    model = ensemble.RandomForestRegressor(**params)
    fit = model.fit(used_x, used_y)
    times['fit_time'] = time.clock() - start

    start = time.clock()
    prediction = fit.predict(x[-validation_size:])
    norm_factor = 1 / (prediction.max() - prediction.min())
    times['prediction_time'] = time.clock() - start

    scores = score_prediction(y[-validation_size:], prediction, norm_factor)

    return {'params': params, 'prediction': prediction, **times, **scores, 'norm_factor': norm_factor}


def evaluate_linear_regression(row, x, y, validation_size):
    """
    Evaluate the model with the params given in the row

    :param row:             The row to get the parameters from
    :param x:               The x input vector
    :param y:               The y target vector
    :param validation_size: The validation distance to use for scoring
    :return:
    """
    used_x = x[:-validation_size]
    used_y = y[:-validation_size]
    params = get_linear_regression_params(row)
    times = dict()

    start = time.clock()
    model = linear_model.LinearRegression(**params)
    fit = model.fit(used_x, used_y)
    times['fit_time'] = time.clock() - start

    start = time.clock()
    prediction = fit.predict(x[-validation_size:])
    norm_factor = 1 / (prediction.max() - prediction.min())
    times['prediction_time'] = time.clock() - start

    scores = score_prediction(y[-validation_size:], prediction, norm_factor)

    return {'params': params, 'prediction': prediction, **times, **scores, 'norm_factor': norm_factor}


def evaluate_gru(row, x, y, validation_size):
    """
    Evaluate the model with the params given in the row

    :param row:             The row to get the parameters from
    :param x:               The x input vector
    :param y:               The y target vector
    :param validation_size: The validation distance to use for scoring
    :return:
    """
    from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

    scaled_x, scaled_y, x_scaler, y_scaler = scale_inputs(x, y)
    scaled_x = scaled_x.values.reshape(scaled_x.shape[0], scaled_x.shape[1], 1)
    scaled_y = scaled_y.values
    used_x = scaled_x[:-validation_size]
    used_y = scaled_y[:-validation_size]
    params = get_gru_params(row)
    times = dict()

    start = time.clock()
    model = KerasRegressor(build_fn=create_gru, **params)
    fit = model.fit(used_x, used_y)
    times['fit_time'] = time.clock() - start

    start = time.clock()
    prediction = rescale_array(model.predict(scaled_x[-validation_size:]), y_scaler)
    norm_factor = 1 / (prediction.max() - prediction.min())
    times['prediction_time'] = time.clock() - start

    scores = score_prediction(y[-validation_size:], prediction, norm_factor)

    return {'params': params, 'prediction': prediction, **times, **scores, 'norm_factor': norm_factor}


def evaluate_ets(row, y, validation_size, rate):
    """
    Evaluate the model with the params given in the row

    :param row:             The row to get the parameters from
    :param y:               The y target vector
    :param validation_size: The validation distance to use for scoring
    :param rate:            The sampling rate for the values in y
    :return:
    """
    used_y = y[:-validation_size]
    periods = int(len(used_y) / rate)
    params = get_ets_params(row)
    times = dict()

    start = time.clock()
    model = ExponentialSmoothing(used_y, trend='add', seasonal_periods=periods, seasonal='add')
    model.params = params
    fit = model.fit()
    times['fit_time'] = time.clock() - start

    start = time.clock()
    prediction = fit.forecast(validation_size)
    norm_factor = 1 / (prediction.max() - prediction.min())
    times['prediction_time'] = time.clock() - start

    scores = score_prediction(y[-validation_size:], prediction, norm_factor)

    return {'params': params, 'prediction': prediction, **times, **scores, 'norm_factor': norm_factor}


def evaluate_arima(row, y, validation_size):
    """
    Evaluate the model with the params given in the row

    :param row:             The row to get the parameters from
    :param y:               The y target vector
    :param validation_size: The validation distance to use for scoring
    :return:
    """
    used_y = y[:-validation_size]
    params = get_arima_params(row)
    times = dict()

    start = time.clock()
    model = ARIMA(**params)
    fit = model.fit(y=used_y)
    times['fit_time'] = time.clock() - start

    start = time.clock()
    prediction = fit.predict(validation_size)
    norm_factor = 1 / (prediction.max() - prediction.min())
    times['prediction_time'] = time.clock() - start

    scores = score_prediction(y[-validation_size:], prediction, norm_factor)

    return {'params': params, 'prediction': prediction, **times, **scores, 'norm_factor': norm_factor}


def evaluate_prophet(y, validation_size, rate):
    """
    Evaluate the model with the params given in the row

    :param y:               The y target vector
    :param validation_size: The validation distance to use for scoring
    :param rate:            The sampling rate for the values in y
    :return:
    """
    used_y = y[:-validation_size]
    times = dict()

    start = time.clock()
    prophet = Prophet().fit(pd.DataFrame(data={
        'ds': used_y.index,
        'y': used_y
    }))
    times['fit_time'] = time.clock() - start

    start = time.clock()
    future = prophet.make_future_dataframe(validation_size, rate)
    complete_prediction = prophet.predict(future)
    prediction = complete_prediction['yhat'][-validation_size:]
    norm_factor = 1 / (prediction.max() - prediction.min())
    times['prediction_time'] = time.clock() - start

    scores = score_prediction(y[-validation_size:], prediction, norm_factor)

    return {'params': {}, 'prediction': prediction, **times, **scores, 'norm_factor': norm_factor}


def evaluate_best_params(resources, results_folder, evaluation_folder, predictions_folder, debugging=False,
                         debug_length=200):
    """
    Runs evaluation of the found best parameters using the matching csvs

    :param resources:           The folder name of the converted eea weatherdata csv files
    :param results_folder:      The folder name in which the parameter estimation results are contained
    :param predictions_folder:  The folder to save predictions to
    :param evaluation_folder:   The folder name to save evaluation results to
    :param debugging:           If debugging mode should be enabled
    :param debug_length:        The maximum amout of items to use in debugging mode
    """

    def evaluate_model(r, data_frame, differenced_frame):
        """
        Evaluate model with parameters from row r

        :param r:                   The row to extract parameters from
        :param data_frame:          The complete dataframe to use
        :param differenced_frame:   The differenced dataframe
        :return:                    (stats, predictions) each as df
        """
        pollutant, distance, differenced, direct, model = r['pollutant'], int(r['distance']), r[
            'differenced'], eval(r['direct']), r['model']
        if direct:
            artificial = eval(r['artificial'])
        else:
            artificial = False

        validation_distance = 24 if rate == 'H' else 7

        if differenced:
            used_df = differenced_frame
        else:
            used_df = data_frame
        series = used_df[pollutant]

        if direct:
            rotated = series
            for j in range(0, distance):
                rotated = rotate_series(rotated)[:-1]
            rest = used_df.drop(columns=[pollutant])

            if artificial:
                steps = 7 if rate == 'D' else 24
                x = create_artificial_features(series[:len(rotated)], rate, steps)[steps:]
                y = rotated[steps:]
            else:
                x = rest[:len(rotated)]
                y = rotated

            if model == 'knn':
                scoring = evaluate_knn(r, x, y, validation_distance)
            elif model == 'decision_tree':
                scoring = evaluate_decision_tree(r, x, y, validation_distance)
            elif model == 'random_forest':
                scoring = evaluate_random_forest(r, x, y, validation_distance)
            elif model == 'linear_regression':
                scoring = evaluate_linear_regression(r, x, y, validation_distance)
            elif model == 'gru':
                scoring = evaluate_gru(r, x, y, validation_distance)
            else:
                scoring = False
        else:
            if model == 'arima':
                scoring = evaluate_arima(r, series, validation_distance)
            elif model == 'ets':
                scoring = evaluate_ets(r, series, validation_distance, rate)
            elif model == 'prophet':
                scoring = evaluate_prophet(series, validation_distance, rate)
            else:
                scoring = False

        new_stats_df = pd.DataFrame()
        new_predictions_df = pd.DataFrame()

        if scoring:
            prediction = scoring['prediction']
            params = scoring['params']
            scoring.pop('prediction')
            scoring.pop('params')
            stats_dict = {'model': model,
                          'direct': direct,
                          'differenced': differenced,
                          'distance': distance,
                          'artificial': artificial,
                          'pollutant': pollutant,
                          **params,
                          **scoring}
            new_stats_df = pd.DataFrame().from_records([stats_dict.values()], columns=stats_dict.keys())

            new_predictions_df = pd.DataFrame(prediction)
            new_predictions_df['model'] = model
            new_predictions_df['direct'] = direct
            new_predictions_df['differenced'] = differenced
            new_predictions_df['distance'] = distance
            new_predictions_df['artificial'] = artificial
            new_predictions_df['pollutant'] = pollutant

        return new_stats_df, new_predictions_df

    evaluated =  glob.glob(f'{evaluation_folder}/*.csv')
    evaluated_pairs = []
    for csv in evaluated:
        station, rate = csv.replace(f'{evaluation_folder}/', '').replace('.csv', '').replace('day', 'D').replace('hour',
                                                                                                         'H').split('-')
        evaluated_pairs.append((station, rate))

    items = glob.glob(f'{resources}/*.csv')
    for i in range(len(items)):
        csv = items[i]
        station, rate = csv.replace(f'{resources}/', '').replace('.csv', '').replace('day', 'D').replace('hour',
                                                                                                         'H').split('-')

        if (station, rate) in evaluated_pairs:
            print(f'skipping station {station} with rate {rate} because it was already evaluated')
            continue
        print(f'({i + 1}/{len(items)})')

        results = glob.glob(f'{results_folder}/{station}-{rate}*.csv')
        data_df = pd.read_csv(csv, index_col=0, parse_dates=[0], infer_datetime_format=True).drop(
            columns=['AirQualityStationEoICode', 'AveragingTime'])

        if len(data_df > 8760):
            data_df = data_df[:8760]

        if debugging:
            data_df = data_df[:debug_length]

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
        predictions_df = pd.DataFrame()

        eval_results = Parallel(n_jobs=-1)(
            delayed(evaluate_model)(row, data_df, differenced_df) for idx, row in stats_df.iterrows())

        stats, predictions = zip(*eval_results)

        for i in range(len(stats)):
            best_stats_df = pd.concat([best_stats_df, stats[i]])
            predictions_df = pd.concat([predictions_df, predictions[i]])

        best_stats_df = best_stats_df.reset_index()
        best_stats_df = best_stats_df.drop(columns=['index'])

        predictions_df = predictions_df.reset_index()
        predictions_df = predictions_df.rename(index=str, columns={'index': 'step', 0: 'value'})

        best_stats_df.to_csv(f'{evaluation_folder}/{station}-{rate}.csv')
        predictions_df.to_csv(f'{predictions_folder}/{station}-{rate}.csv')
        evaluated_pairs.append((station, rate))


if __name__ == '__main__':
    debug = (not sys.gettrace() is None) | (sys.argv[1] == 'debug') | (sys.argv[1] == True)
    if debug:
        print('Running in debug mode, amount of used items will be limited.')

    resource_loc = 'post'
    results_loc = 'results'
    evaluation_loc = 'eval'
    predictions_loc = 'predictions'

    if not os.path.exists(f'./{evaluation_loc}'):
        os.makedirs(f'./{evaluation_loc}')

    if not os.path.exists(f'./{predictions_loc}'):
        os.makedirs(f'./{predictions_loc}')

    models = ['knn', 'decision_tree', 'random_forest', 'linear_regression', 'gru', 'ets', 'arima', 'prophet']
    evaluate_best_params(resource_loc, results_loc, evaluation_loc, predictions_loc, debug)
