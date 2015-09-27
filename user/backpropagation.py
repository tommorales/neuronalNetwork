__author__ = 'tmorales'

import numpy as np

from src.NeuronalNetwork import NeuronalNetwork
from src.plotData import *

# dataSet
x = np.linspace(-7,7,20)
y = np.sin(x) * 0.5
# plotting the dataSet
plot_dataset(x, y, xLabel="X", yLabel="y", legend="sin(x)")

# Network architecture
input = x.reshape(len(x), 1)
target = y.reshape(len(x), 1)
inputL = [[-7,7]]
hiddenOutputL = [7,7,7,1]

# Fit and prediction
nn = NeuronalNetwork(input, target, inputL, hiddenOutputL)
prediction, error = nn.run()

# Plotting results
plot_error(x, y, prediction, error, title="Backpropagation Algoritm", xLabel="Epoch", yLabel="error (default SSE)")
