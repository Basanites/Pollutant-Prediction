from abc import ABC, abstractmethod
from sklearn import neighbors, ensemble, tree


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
    def predict(self):
        pass


class ARIMAPredictor(Predictor):
    def predict(self):
        pass


class ProphetPredictor(Predictor):
    def predict(self):
        pass


class LSTMPredictor(Predictor):
    def predict(self):
        pass
