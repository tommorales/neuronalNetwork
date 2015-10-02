__author__ = 'tmorales'

import neurolab as nl


class NeuronalNetwork(object):
    """

    """
    def __init__(self, input, target, inputL, hiddenOutputL):
        self.input = input
        self.target = target
        self.inputL = inputL
        self.hiddenOutputL = hiddenOutputL

    def network(self):
        self.net = nl.net.newff(self.inputL, self.hiddenOutputL)

    def fit(self):
        self.net.trainf = nl.train.train_gd
        self.err = self.net.train(self.input, self.target, show=30, adapt=True)
        return self.err

    def predict(self):
        self.out = self.net.sim(self.input)
        return self.out

    def run(self):
        self.network()
        error = self.fit()
        prediction = self.predict()
        return prediction, error

# ************************************************************************************
# ************************************************************************************
# ************************************************************************************
class NeuronalNetwork2():
    def __init__(self, inputTrain, targetTrain, inputTest, inputL, hiddenOutputL):
        self.inputTrain = inputTrain
        self.targetTrain = targetTrain
        self.inputTest = inputTest
        self.inputL = inputL
        self.hiddenOutputL = hiddenOutputL

    def _network(self):
        raise NotImplementedError

    def _fit(self):
        raise NotImplementedError

    def _predict(self):
        raise NotImplementedError

    def run(self, train="gd", error="SSE"):
        self._network()
        error = self._fit(train=train, error=error)
        prediction = self._predict()
        return prediction, error



class Multilayer(NeuronalNetwork2):

    def _network(self):
        self.net = nl.net.newff(self.inputL, self.hiddenOutputL)

    def _fit(self, train="gd", error="SSE"):
        # Optimation algorithm ********************
        if train is "gd": self.net.trainf = nl.train.train_gd   # gradient descent backpropagation
        if train is "gdm": pass                                 # gradient descente with momento bp.
        if train is "gda": pass                                 # g. d. with adaptative learning range.
        # Error *****************************************************************
        if error is "CEE": self.net.errorf = nl.error.CCE()   # Cross-Entropy error function
        if error is "MAE": self.net.errorf = nl.error.MAE()   # Mean absolute error function
        if error is "MSE": self.net.errorf = nl.error.MSE()   # Mean Squared error function
        if error is "SAE": self.net.errorf = nl.error.SAE()   # Sum absolute error function
        if error is "SSE": pass                               # Sum square error function
        # Trainning *************************************************************
        self.err = self.net.train(self.inputTrain, self.targetTrain, show=500, adapt=True)
        return self.err

    def _predict(self):
        self.out = self.net.sim(self.inputTest)
        return self.out


