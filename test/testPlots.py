__author__ = 'tmorales'

import numpy as np

from src.plotData import plot_dataset, plot_error


def dataSet():
    x = np.linspace(-7, 7, 200)
    y = np.sin(x)

    plot_dataset(x,y, title="Sin Function", xLabel="X", yLabel="Y", legend="sin(x)")

#dataSet()


def error():
    x = np.linspace(-7, 7, 200)
    y = np.sin(x)
    prediction = np.sin(x+0.5)
    error = range(len(x))

    plot_error(x, y, prediction, error, title="Backpropagation Algoritm", xLabel="Epoch", yLabel="error (default SSE)")


error()