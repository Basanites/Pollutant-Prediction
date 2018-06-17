from abc import ABC, abstractmethod
import pandas as pd
from sklearn import neighbors, ensemble, tree
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from fbprophet import Prophet


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
    def __init__(self, traindata_x, traindata_y, testdata_x, testdata_y, n_estimators):
        super(RandomForestPredictor, self).__init__(traindata_x, traindata_y, testdata_x, testdata_y)
        self.model = ensemble.RandomForestRegressor(n_estimators=n_estimators).fit(self.train['x'], self.train['y'])

    def predict(self):
        self.y_ = self.model.predict(self.test['x'])
        return self.y_


class KNearestNeighborsPredictor(Predictor):
    def __init__(self, traindata_x, traindata_y, testdata_x, testdata_y, n_neighbors, weights):
        super(KNearestNeighborsPredictor, self).__init__(traindata_x, traindata_y, testdata_x, testdata_y)
        self.model = neighbors.KNeighborsRegressor(n_neighbors, weights=weights).fit(self.train['x'], self.train['y'])

    def predict(self):
        self.y_ = self.model.predict(self.test['x'])
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
