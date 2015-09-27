__author__ = 'tmorales'

import numpy as np

from src.NeuronalNetwork import NeuronalNetwork

def test_NeuronalNetwork():
    # DataSet
    x = np.linspace(-7,7,20)
    y = np.sin(x) * 0.5

    input = x.reshape(len(x), 1)
    target = y.reshape(len(x), 1)
    inputL = [[-7,7]]
    hiddenOutputL = [7,7,7,1]

    nn = NeuronalNetwork(input, target, inputL, hiddenOutputL)
    prediction, error = nn.run()
    print ""
    print prediction
    print ""
    print error

test_NeuronalNetwork()


