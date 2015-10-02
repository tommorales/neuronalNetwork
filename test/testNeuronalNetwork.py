__author__ = 'tmorales'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.NeuronalNetwork import NeuronalNetwork
from src.NeuronalNetwork import Multilayer


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

#test_NeuronalNetwork()




def multiplayer():
    # DataSet
    x = np.linspace(-7,7,20)
    y = np.sin(x) * 0.5

    input = x.reshape(len(x), 1)
    target = y.reshape(len(x), 1)
    inputL = [[-7,7]]
    hiddenOutputL = [7,7,7,1]

    nn = Multilayer(input, target, inputL, hiddenOutputL)
    prediction, error_CCE = nn.run(error="CCE")
    prediction, error_MAE = nn.run(error="MAE")
    prediction, error_SSE = nn.run(error="MSE")
    prediction, error_SAE = nn.run(error="SAE")
    prediction, error_SSE = nn.run()


    pd.DataFrame({"CCE": error_CCE,
                  "MAE": error_MAE,
                  "SAE": error_SAE,
    #                  "SSE": error_SSE
                  }).plot()
    plt.show()


#multiplayer()

from sklearn.datasets import make_blobs
from sklearn.cross_validation import train_test_split

def multiplayer2():

    # Network ************************************************************
    inputL = [[-7,7]]
    hiddenOutputL = [7,1]
    # ********************************************************************
    #nn = Multilayer(X_train, y_train, X_train, inputL, hiddenOutputL)
    #prediction, error_SSE = nn.run()


multiplayer2()