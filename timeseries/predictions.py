from abc import ABC, abstractmethod
import pandas as pd
from sklearn import neighbors, ensemble, tree
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from fbprophet import Prophet
import calendar


class Predictor(ABC):
    def __init__(self, traindata_x, traindata_y, testdata_x, testdata_y):
        self.train = {'x': traindata_x, 'y': traindata_y}
        self.test = {'x': testdata_x, 'y': testdata_y}
        self.model = None
        self.y_ = None

    @abstractmethod
    def predict(self):
        pass


class LinearRegressionPredictor(Predictor):
    def predict(self):
        pass


class DecisionTreePredictor(Predictor):
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
        super(DecisionTreePredictor, self).__init__(traindata_x, traindata_y, testdata_x, testdata_y)
        self.type = mode.lower()
        self.steps = steps

        if self.type == 'multimodel':
            self.models = list()
            train_y = self.train['y']
            for i in range(0, steps):
                self.models[i] = tree.DecisionTreeRegressor(max_depth=depth).fit(self.train['x'][:-i], train_y[:i - 1])
                train_y[:] = train_y[1:] + [train_y[0]]
        else:
            self.model = tree.DecisionTreeRegressor(max_depth=depth).fit(self.train['x'], self.train['y'])

    def predict(self):
        """
        :return: the predicted values according to set inputs as list
        """
        if self.type is not None and self.steps > 1:
            if self.type == 'recursive':
                return self._recursive_predict()
            if self.type == 'multimodel':
                return self._multimodel_predict()
        self.y_ = self.model.predict(self.test['x'])
        return self.y_

    def _recursive_predict(self):
        """
        Recursively predicts the predictors set number of steps further than test input is given.
        For all given input values no recursion is used.

        :return: the recursive prediction values
        """
        inputs = list()
        inputs[0] = self.test['x'][-1]
        self.y_ = self.model.predict(self.test['x'][:-1])
        for i in range(0, self.steps):
            prediction = self.model.predict(inputs[i])
            inputs.append(prediction)
            self.y_.append(prediction)
        return self.y_

    def _multimodel_predict(self):
        """
        Predicts the predictors set number of steps further than test input is given.
        Uses a different model for each further step ahead.
        Only the additional steps are predicted using shifted models.

        :return: the multimodel prediction values
        """
        self.y_ = self.models[0].predict(self.test['x'][:-1])
        for i, model in self.models:
            self.y_ = model.predict(self.test['x'][-1])
        return self.y_


class RandomForestPredictor(Predictor):
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
        super(RandomForestPredictor, self).__init__(traindata_x, traindata_y, testdata_x, testdata_y)
        self.type = mode.lower()
        self.steps = steps

        if self.type == 'multimodel':
            self.models = list()
            train_y = self.train['y']
            for i in range(0, steps):
                self.models[i] = ensemble.RandomForestRegressor(n_estimators=n_estimators) \
                    .fit(self.train['x'][:-i], train_y[:i - 1])
                train_y[:] = train_y[1:] + [train_y[0]]
        else:
            self.model = ensemble.RandomForestRegressor(n_estimators=n_estimators).fit(self.train['x'], self.train['y'])

    def predict(self):
        """
        :return: the predicted values according to set inputs as list
        """
        if self.type is not None and self.steps > 1:
            if self.type == 'recursive':
                return self._recursive_predict()
            if self.type == 'multimodel':
                return self._multimodel_predict()
        self.y_ = self.model.predict(self.test['x'])
        return self.y_

    def _recursive_predict(self):
        """
        Recursively predicts the predictors set number of steps further than test input is given.
        For all given input values no recursion is used.

        :return: the recursive prediction values
        """
        inputs = list()
        inputs[0] = self.test['x'][-1]
        self.y_ = self.model.predict(self.test['x'][:-1])
        for i in range(0, self.steps):
            prediction = self.model.predict(inputs[i])
            inputs.append(prediction)
            self.y_.append(prediction)
        return self.y_

    def _multimodel_predict(self):
        """
        Predicts the predictors set number of steps further than test input is given.
        Uses a different model for each further step ahead.
        Only the additional steps are predicted using shifted models.

        :return: the multimodel prediction values
        """
        self.y_ = self.models[0].predict(self.test['x'][:-1])
        for i, model in self.models:
            self.y_ = model.predict(self.test['x'][-1])
        return self.y_


class KNearestNeighborsPredictor(Predictor):
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
        super(KNearestNeighborsPredictor, self).__init__(traindata_x, traindata_y, testdata_x, testdata_y)
        self.type = mode.lower()
        self.steps = steps

        if self.type == 'multimodel':
            self.models = list()
            train_y = self.train['y']
            for i in range(0, steps):
                self.models[i] = neighbors.KNeighborsRegressor(n_neighbors, weights=weights) \
                    .fit(self.train['x'][:-i], train_y[:i - 1])
                train_y[:] = train_y[1:] + [train_y[0]]
        else:
            self.model = neighbors.KNeighborsRegressor(n_neighbors, weights=weights).fit(self.train['x'],
                                                                                         self.train['y'])

    def predict(self):
        """
        :return: the predicted values according to set inputs as list
        """
        if self.type is not None and self.steps > 1:
            if self.type == 'recursive':
                return self._recursive_predict()
            if self.type == 'multimodel':
                return self._multimodel_predict()
        self.y_ = self.model.predict(self.test['x'])
        return self.y_

    def _recursive_predict(self):
        """
        Recursively predicts the predictors set number of steps further than test input is given.
        For all given input values no recursion is used.

        :return: the recursive prediction values
        """
        inputs = list()
        inputs[0] = self.test['x'][-1]
        self.y_ = self.model.predict(self.test['x'][:-1])
        for i in range(0, self.steps):
            prediction = self.model.predict(inputs[i])
            inputs.append(prediction)
            self.y_.append(prediction)
        return self.y_

    def _multimodel_predict(self):
        """
        Predicts the predictors set number of steps further than test input is given.
        Uses a different model for each further step ahead.
        Only the additional steps are predicted using shifted models.

        :return: the multimodel prediction values
        """
        self.y_ = self.models[0].predict(self.test['x'][:-1])
        for i, model in self.models:
            self.y_ = model.predict(self.test['x'][-1])
        return self.y_


class ETSPredictor(Predictor):
    """
    Provides functionality to predict outputs by learning the training data using ETS
    """

    def __init__(self, traindata_x, traindata_y, testdata_x, seasonlength: int, trendtype='additive',
                 seasontype='additive', testdata_y=None):
        """
        Initializes the ETS Predictor object

        :param traindata_x:     training x vector
        :param traindata_y:     training y vector
        :param testdata_x:      testing x vector (the values to predict from)
        :param trendtype:       type of ETS trend component. One of 'multiplicative' and 'additive'. Default 'additive'
        :param seasontype:      type of ETS season component. One of 'multiplicative' and 'additive'. Default 'additive'
        :param seasonlength:    the length to use for the season according to the given data
        :param testdata_y:      testing y vector (only used when testing model accuracy)
        """
        super(ETSPredictor, self).__init__(traindata_x, traindata_y, testdata_x, testdata_y)
        self.model = ExponentialSmoothing(self.train['y'],
                                          trend=trendtype,
                                          seasonal=seasontype,
                                          seasonal_periods=seasonlength).fit(optimized=True)

    def predict(self, steps: int):
        """
        Predicts n future steps from the given training data

        :param steps:   how many steps to predict
        :return:        the values of the forecast steps using the same period as the training data
        """
        self.y_ = self.model.forecast(steps)
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
        super(ARIMAPredictor, self).__init__(traindata_x, traindata_y, testdata_x, testdata_y)
        self.model = ARIMA(traindata_x, order=order)

    def predict(self):
        """
        Predicts n future steps from the end of training data

        :return: the values of the forecast steps
        """
        self.model = self.model.fit(disp=0)
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
        super(ProphetPredictor, self).__init__(traindata_x, traindata_y, testdata_x, testdata_y)
        self.model = Prophet().fit(pd.DataFrame(data={'ds': self.train['x'], 'y': self.train['y']}))

    def predict(self, steps: int):
        """
        Predicts n steps from the end of training data

        :param steps:   the amount of steps to predict further
        :return:        the predicted values
        """
        future = self.model.make_future_dataframe(periods=steps, freq=self.train['x'].infer_freq(warn=False))
        self.y_ = self.model.predict(future)['yhat']
        return self.y_


class LSTMPredictor(Predictor):
    def predict(self):
        pass


def create_artificial_features(series, frequency='H', steps=7):
    """
    Creates artificial features for a given series

    :param series:      the base series to use
    :param frequency:   the frequency of values in the series
    :param steps:       the amount of steps to lag the series by
    :return:            the dataframe containing the artificial features for the input series
    """
    lagged = create_lagged_features(series, frequency, steps)

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
