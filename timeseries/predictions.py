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
    def __init__(self, traindata_x, traindata_y, testdata_x, testdata_y, depth):
        super(DecisionTreePredictor, self).__init__(traindata_x, traindata_y, testdata_x, testdata_y)
        self.model = tree.DecisionTreeRegressor(max_depth=depth).fit(self.train['x'], self.train['y'])

    def predict(self):
        self.y_ = self.model.predict(self.test['x'])
        return self.y_


class RandomForestPredictor(Predictor):
    """
    Provides functionality to predict the output values by learning the training data
    """

    def __init__(self, traindata_x, traindata_y, testdata_x, n_estimators, testdata_y=None, type=None, steps=1):
        """
        Initializes the Random Forest Predictor

        :param traindata_x:     training x vector
        :param traindata_y:     training y vector
        :param testdata_x:      testing x vector (the values to predict from)
        :param n_estimators:    the number of estimators to use
        :param testdata_y:      testing y vector (only used when testing model accuracy)
        :param type:            one of 'multimodel' and 'recursive' for multistep forecast
        :param steps:           only used when type is set. Number of steps for multistep forecast
        """
        super(RandomForestPredictor, self).__init__(traindata_x, traindata_y, testdata_x, testdata_y)
        self.type = type.lower()
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
    Provides functionality to predict values from given inputs after having learned from training data.
    """

    def __init__(self, traindata_x, traindata_y, testdata_x, n_neighbors, weights, testdata_y=None, type=None,
                 steps=1):
        """
        Initializes the KNN Predictor

        :param traindata_x: training x vector
        :param traindata_y: training y vector
        :param testdata_x:  testing x vector (the values to predict from)
        :param n_neighbors: how many neighbors to use for knn
        :param weights:     one of TODO
        :param testdata_y:  testing y vector (only used when testing model accuracy)
        :param type:        one of 'multimodel' and 'recursive' for multistep forecast
        :param steps:       only used when type is set. Number of steps for multistep forecast
        """
        super(KNearestNeighborsPredictor, self).__init__(traindata_x, traindata_y, testdata_x, testdata_y)
        self.type = type.lower()
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
    def __init__(self, traindata_x, traindata_y, testdata_x, testdata_y, trendtype, seasontype, seasonlength: int):
        super(ETSPredictor, self).__init__(traindata_x, traindata_y, testdata_x, testdata_y)
        self.model = ExponentialSmoothing(self.train['y'],
                                          trend=trendtype,
                                          seasonal=seasontype,
                                          seasonal_periods=seasonlength).fit(optimized=True)

    def predict(self, steps: int):
        self.y_ = self.model.forecast(steps)
        return self.y_


class ARIMAPredictor(Predictor):
    def __init__(self, traindata_x, traindata_y, testdata_x, testdata_y, order):
        super(ARIMAPredictor, self).__init__(traindata_x, traindata_y, testdata_x, testdata_y)
        self.model = ARIMA(traindata_x, order=order)

    def predict(self):
        self.model = self.model.fit(disp=0)
        pass


class ProphetPredictor(Predictor):
    def __init__(self, traindata_x, traindata_y, testdata_x, testdata_y):
        super(ProphetPredictor, self).__init__(traindata_x, traindata_y, testdata_x, testdata_y)
        self.model = Prophet().fit(pd.DataFrame(data={'ds': self.train['x'], 'y': self.train['y']}))

    def predict(self, steps: int):
        future = self.model.make_future_dataframe(periods=steps, freq=self.train['x'].infer_freq(warn=False))
        self.y_ = self.model.predict(future)['yhat']
        return self.y_


class LSTMPredictor(Predictor):
    def predict(self):
        pass


def create_artificial_features(series, frequency='H', steps=7):
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
    lagged = pd.DataFrame()

    for i in range(1, steps + 1):
        lagged['lag {}{}'.format(i, frequency)] = series.shift(i, freq=frequency)

    lagged.index = series.index
    lagged = lagged[steps + 1:]

    return lagged.interpolate()
