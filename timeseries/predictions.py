import calendar
import math
import time

import pandas as pd
from fbprophet import Prophet
from sklearn import neighbors, ensemble, tree, linear_model
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def _calc_mae(expected, actual):
    mae = 0
    for i in range(0, len(expected)):
        mae += math.fabs(expected.iloc[i] - actual[i])
    return mae / len(expected)


def _calc_mse(expected, actual):
    mse = 0
    for i in range(0, len(expected)):
        mse += (expected.iloc[i] - actual[i]) ** 2
    return mse / len(expected)


class Predictor():
    def __init__(self, traindata_x, traindata_y, testdata_x, testdata_y, mode='single', steps=1):
        start = time.time()
        self.train = {'x': traindata_x, 'y': traindata_y}
        self.test = {'x': testdata_x, 'y': testdata_y}
        self.model = None
        self.y_ = list()
        self.y_s = list()
        self.type = mode.lower(),
        self.steps = steps
        self.models = list()
        self.time = dict()
        self.time['init'] = time.time() - start

    def get_prediction_stats(self):
        stats = dict()
        if not len(self.y_):
            self.predict()

        if self.type != 'single':
            stats['mse'] = self.get_multistep_mse()
            stats['rmse'] = self.get_multistep_rmse()
            stats['mae'] = self.get_multistep_mae()
        else:
            stats['mse'] = self.get_mse()
            stats['rmse'] = self.get_rmse()
            stats['mae'] = self.get_mae()
        # needs to be subclass based because fits are handled differently
        # stats['residual mse'] = self.get_residual_mse()
        # stats['residual rmse'] = self.get_residual_rmse()
        # stats['residual mae'] = self.get_residual_mae()
        stats['initialization_time'] = self.get_initialization_time()
        stats['prediction_time'] = self.get_prediction_time()
        stats['complete_time'] = self.get_prediction_time() + self.get_initialization_time()
        stats['steps'] = self.steps

        return stats

    def predict(self):
        """
        :return:    the predicted values according to set inputs as list
                    list of lists for multimodel and recursive
        """
        start = time.time()

        if self.type is not None and self.steps > 1:
            if self.type == 'recursive':
                self.time['predict'] = time.time() - start
                return self._recursive_predict()
            if self.type == 'multimodel':
                self.time['predict'] = time.time() - start
                return self._multimodel_predict()
        self.y_ = self.model.predict(self.test['x'])

        self.time['predict'] = time.time() - start
        return self.y_

    def _recursive_predict(self):
        """
        Recursively predicts the predictors set number of steps further than test input is given.
        For all given input values no recursion is used.

        :return: the list of recursive prediction values for timestep: list[model][step]
        """
        # inputs = list()
        # for i in range(0, len(self.test['x'])):
        #     inputs[0] = self.test['x'].iloc[i]
        #     for j in range(0, self.steps):
        #         prediction = self.model.predict(inputs[j])
        #         self.y_s[i].append(prediction)
        #
        #         current_x =
        #
        #         object[f'lag_{i}{freq}'] = object[f'lag_{i - 1}{freq}']
        #         inputs.append(object)
        # return self.y_s
        pass

    def _multimodel_predict(self):
        """
        Predicts the predictors set number of steps further than test input is given.
        Uses a different model for each further step ahead.

        :return: the list of multimodel prediction values for timestep: list[model][step]
        """

        for i in range(0, len(self.models)):
            self.y_s[i] = self.models[i].predict(self.test['x'])
        return self.y_s

    def get_mse(self):
        return _calc_mse(self.test['y'][:len(self.y_)], self.y_)

    def get_multistep_mse(self):
        out = list()
        for i in range(0, len(self.y_s)):
            y_ = self.y_s[i]
            out.append(_calc_mse(self.test['y'][i + 1:len(y_)], y_[:-(i+1)]))
        return out

    def get_rmse(self):
        return self.get_mse() ** 0.5

    def get_multistep_rmse(self):
        out = list()
        for i in range(0, len(self.y_s)):
            y_ = self.y_s[i]
            out.append(_calc_mse(self.test['y'][i + 1:len(y_)], y_[:-(i+1)]) ** 0.5)
        return out

    def get_mae(self):
        return _calc_mae(self.test['y'][:len(self.y_)], self.y_)

    def get_multistep_mae(self):
        out = list()
        for i in range(0, len(self.y_s)):
            y_ = self.y_s[i]
            out.append(_calc_mae(self.test['y'][i + 1:len(y_)], y_[:-(i+1)]))
        return out

    def get_initialization_time(self):
        return self.time['init']

    def get_prediction_time(self):
        return self.time['predict']

class SingleStepPredictor(Predictor):
    def get_prediction_stats(self):
        stats = super(SingleStepPredictor, self).get_prediction_stats()
        stats['mode'] = self.type
        if self.type == 'single':
            stats['steps'] = 1
        return stats


class LinearRegressionPredictor(SingleStepPredictor):
    """
    Provides functionality to train and predict using a Linear Regression Model
    """

    def __init__(self, traindata_x, traindata_y, testdata_x, testdata_y=None, mode=None, steps=1):
        """
        Initializes the Linear Regression Predictor object

        :param traindata_x: training x vector
        :param traindata_y: training y vector
        :param testdata_x:  testing x vector (the values to predict from)
        :param testdata_y:  testing y vector (only used when testing model accuracy)
        :param mode:        one of 'multimodel' and 'recursive' for multistep forecast
        :param steps:       only used when mode is set. Number of steps for multistep forecast
        """
        start = time.time()

        super(LinearRegressionPredictor, self).__init__(traindata_x, traindata_y, testdata_x, testdata_y, mode, steps)

        if self.type == 'multimodel':
            train_y = self.train['y']
            self.models.append(linear_model.LinearRegression().fit(self.train['x'], train_y))
            for i in range(1, steps + 1):
                # y values are rotated right for further predictions
                train_y[:] = train_y[1:] + [train_y[0]]
                self.models.append(linear_model.LinearRegression().fit(self.train['x'][:-i], train_y[:-i]))
        else:
            self.model = linear_model.LinearRegression().fit(self.train['x'], self.train['y'])

        self.time['init'] = time.time() - start


class DecisionTreePredictor(SingleStepPredictor):
    """
    Provides functionality to predict outputs by learning the training data using a decision tree.
    """

    def __init__(self, traindata_x, traindata_y, testdata_x, depth, testdata_y=None, mode=None, steps=1):
        """
        Initializes the Decision Tree Predictor object

        :param traindata_x: training x vector
        :param traindata_y: training y vector
        :param testdata_x:  testing x vector (the values to predict from)
        :param depth:       the maximum depth to use for the tree
        :param testdata_y:  testing y vector (only used when testing model accuracy)
        :param mode:        one of 'multimodel' and 'recursive' for multistep forecast
        :param steps:       only used when mode is set. Number of steps for multistep forecast
        """
        start = time.time()

        super(DecisionTreePredictor, self).__init__(traindata_x, traindata_y, testdata_x, testdata_y)

        self.type = mode.lower()
        self.steps = steps
        self.depth = depth

        if self.type == 'multimodel':
            train_y = self.train['y']
            self.models.append(tree.DecisionTreeRegressor(max_depth=depth).fit(self.train['x'], train_y))
            for i in range(1, steps + 1):
                # y values are rotated right for further predictions
                train_y[:] = train_y[1:] + [train_y[0]]
                self.models.append(tree.DecisionTreeRegressor(max_depth=depth).fit(self.train['x'][:-i], train_y[:-i]))
        else:
            self.model = tree.DecisionTreeRegressor(max_depth=depth).fit(self.train['x'], self.train['y'])

        self.time['init'] = time.time() - start

    def get_prediction_stats(self):
        stats = super(DecisionTreePredictor, self).get_prediction_stats()
        stats['depth'] = self.depth
        return stats


class RandomForestPredictor(SingleStepPredictor):
    """
    Provides functionality to predict the output values by learning the training data by using a random forest.
    """

    def __init__(self, traindata_x, traindata_y, testdata_x, n_estimators, testdata_y=None, mode=None, steps=1):
        """
        Initializes the Random Forest Predictor

        :param traindata_x:     training x vector
        :param traindata_y:     training y vector
        :param testdata_x:      testing x vector (the values to predict from)
        :param n_estimators:    the number of estimators to use
        :param testdata_y:      testing y vector (only used when testing model accuracy)
        :param mode:            one of 'multimodel' and 'recursive' for multistep forecast
        :param steps:           only used when mode is set. Number of steps for multistep forecast
        """
        start = time.time()

        super(RandomForestPredictor, self).__init__(traindata_x, traindata_y, testdata_x, testdata_y)

        self.type = mode.lower() if mode else 'single'
        self.steps = steps
        self.n_estimators = n_estimators

        if self.type == 'multimodel':
            train_y = self.train['y']
            self.models.append(ensemble.RandomForestRegressor(n_estimators=n_estimators).fit(self.train['x'], train_y))
            for i in range(1, steps + 1):
                # y values are rotated right for further predictions
                train_y[:] = train_y[1:] + [train_y[0]]
                self.models.append(
                    ensemble.RandomForestRegressor(n_estimators=n_estimators).fit(self.train['x'][:-i], train_y[:-i]))
        else:
            self.model = ensemble.RandomForestRegressor(n_estimators=n_estimators).fit(self.train['x'], self.train['y'])

        self.time['init'] = time.time() - start

    def get_prediction_stats(self):
        stats = super(RandomForestPredictor, self).get_prediction_stats()
        stats['n_estimators'] = self.n_estimators
        return stats


class KNearestNeighborsPredictor(SingleStepPredictor):
    """
    Provides functionality to predict values from given inputs after having learned from training data by using KNN.
    """

    def __init__(self, traindata_x, traindata_y, testdata_x, n_neighbors, weights, testdata_y=None, mode=None,
                 steps=1):
        """
        Initializes the KNN Predictor

        :param traindata_x: training x vector
        :param traindata_y: training y vector
        :param testdata_x:  testing x vector (the values to predict from)
        :param n_neighbors: how many neighbors to use for knn
        :param weights:     one of TODO
        :param testdata_y:  testing y vector (only used when testing model accuracy)
        :param mode:        one of 'multimodel' and 'recursive' for multistep forecast
        :param steps:       only used when mode is set. Number of steps for multistep forecast
        """
        start = time.time()

        super(KNearestNeighborsPredictor, self).__init__(traindata_x, traindata_y, testdata_x, testdata_y)

        self.type = mode.lower()
        self.steps = steps
        self.n_neighbors = n_neighbors

        if self.type == 'multimodel':
            train_y = self.train['y']
            self.models.append(
                neighbors.KNeighborsRegressor(n_neighbors, weights=weights).fit(self.train['x'], train_y))
            for i in range(1, steps + 1):
                train_y[:] = train_y[1:] + [train_y[0]]
                self.models.append(
                    neighbors.KNeighborsRegressor(n_neighbors, weights=weights).fit(self.train['x'][:-i], train_y[:-i]))
        else:
            self.model = neighbors.KNeighborsRegressor(n_neighbors, weights=weights).fit(self.train['x'],
                                                                                         self.train['y'])
        self.time['init'] = time.time() - start

    def get_prediction_stats(self):
        stats = super(KNearestNeighborsPredictor, self).get_prediction_stats()
        stats['n_neighbors'] = self.n_neighbors
        return stats


class ETSPredictor(Predictor):
    """
    Provides functionality to predict outputs by learning the training data using ETS
    """

    def __init__(self, traindata_x, traindata_y, testdata_x, seasonlength: int, trendtype='additive',
                 seasontype='additive', steps=1, testdata_y=None):
        """
        Initializes the ETS Predictor object

        :param traindata_x:     training x vector
        :param traindata_y:     training y vector
        :param testdata_x:      testing x vector (the values to predict from)
        :param trendtype:       type of ETS trend component. One of 'multiplicative' and 'additive'. Default 'additive'
        :param seasontype:      type of ETS season component. One of 'multiplicative' and 'additive'. Default 'additive'
        :param seasonlength:    the length to use for the season according to the given data
        :param steps:           amount of timesteps to forecast for
        :param testdata_y:      testing y vector (only used when testing model accuracy)
        """
        start = time.time()

        super(ETSPredictor, self).__init__(traindata_x, traindata_y, testdata_x, testdata_y)
        self.model = ExponentialSmoothing(self.train['y'],
                                          trend=trendtype,
                                          seasonal=seasontype,
                                          seasonal_periods=seasonlength).fit(optimized=True)
        self.steps = steps

        self.time['init'] = time.time() - start

    def predict(self):
        """
        Predicts n future steps from the given training data

        :return:        the values of the forecast steps using the same period as the training data
        """
        start = time.time()

        self.y_ = self.model.forecast(self.steps)

        self.time['predict'] = time.time() - start
        return self.y_


class ARIMAPredictor(Predictor):
    """
    Provides functionality to predict from training data using ARIMA model
    """

    def __init__(self, traindata_x, traindata_y, testdata_x, order, testdata_y=None):
        """
        Initializes the ARIMA Predictor object

        :param traindata_x: training x vector
        :param traindata_y: training y vector
        :param testdata_x:  testing x vector (the values to predict from)
        :param order:       TODO triple? of the pdq values for ARIMA
        :param testdata_y:  testing y vector (only used when testing model accuracy)
        """
        start = time.time()

        super(ARIMAPredictor, self).__init__(traindata_x, traindata_y, testdata_x, testdata_y)
        self.model = ARIMA(traindata_x, order=order)

        self.time['init'] = time.time() - start

    def predict(self):
        """
        Predicts n future steps from the end of training data

        :return: the values of the forecast steps
        """
        start = time.time()

        self.model = self.model.fit(disp=0)

        self.time['predict'] = time.time() - start
        pass


class ProphetPredictor(Predictor):
    """
    Provides functionality to predict from training data using FBProphet
    """

    def __init__(self, traindata_x, traindata_y, testdata_x, testdata_y=None):
        """
        Initializes the Prophet Predictor object
        :param traindata_x: training x vector
        :param traindata_y: training y vector
        :param testdata_x:  testing x vector (the values to predict from)
        :param testdata_y:  testing y vector (only used when testing model accuracy)
        """
        start = time.time()
        super(ProphetPredictor, self).__init__(traindata_x, traindata_y, testdata_x, testdata_y)
        self.model = Prophet().fit(pd.DataFrame(data={'ds': self.train['x'], 'y': self.train['y']}))

        self.time['init'] = time.time() - start

    def predict(self):
        """
        Predicts n steps from the end of training data

        :return:        the predicted values
        """
        start = time.time()

        future = self.model.make_future_dataframe(periods=self.steps, freq=self.train['x'].infer_freq(warn=False))
        self.y_ = self.model.predict(future)['yhat']

        self.time['predict'] = time.time() - start
        return self.y_


class LSTMPredictor(Predictor):
    def predict(self):
        pass


def create_artificial_features(series, frequency='H', steps=7):
    """
    Creates artificial features for a given series with Timestamp Index

    :param series:      the base series to use
    :param frequency:   the frequency of values in the series
    :param steps:       the amount of steps to lag the series by
    :return:            the dataframe containing the artificial features for the input series
    """
    interpolated = series.interpolate(method='time', frequency=frequency)
    lagged = create_lagged_features(interpolated, frequency, steps)

    statistics = lagged
    statistics['sum'] = lagged.sum(axis=1)
    statistics['mean'] = lagged.mean(axis=1)
    statistics['median'] = lagged.median(axis=1)

    weekdays = pd.get_dummies(lagged.index.weekday_name)
    weekdays = weekdays.applymap(lambda x: bool(x))
    weekdays.index = lagged.index

    months = pd.get_dummies(lagged.index.month.map(lambda x: calendar.month_abbr[x]))
    months = months.applymap(lambda x: bool(x))
    months.index = lagged.index

    out = statistics.join(weekdays).join(months)

    return out


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
    lagged = lagged[steps + 1:]

    return lagged.interpolate()
