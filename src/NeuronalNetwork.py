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

    def _network(self):
        self.net = nl.net.newff(self.inputL, self.hiddenOutputL)

    def _fit(self):
        self.net.trainf = nl.train.train_gd
        self.err = self.net.train(self.input, self.target, show=30, adapt=True)
        return self.err

    def _predict(self):
        self.out = self.net.sim(self.input)
        return self.out

    def run(self):
        self._network()
        error = self._fit()
        prediction = self._predict()
        return prediction, error
