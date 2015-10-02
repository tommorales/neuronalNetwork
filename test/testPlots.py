__author__ = 'tmorales'

import numpy as np

from src.plotData import *


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


#error()


def twoHorizontalPlots():
    x = np.linspace(-7, 7, 200)
    y = np.sin(x)
    y2 = np.sin(x)+2
    plot_twoHorizontalPlots(x, y, x, y2,
                            title1="Titulo 1", xLabel1="X1", yLabel1="Y1", legend1="sin(x)",
                            title2="Titulo 2", xLabel2="X2", yLabel2="Y2", legend2="sin(x)*0.5"
                            )

#twoHorizontalPlots()

def train_test_data():
    # Line ****************************
    x = np.linspace(-7, 7, 200)
    y = np.sin(x)
    y2 = np.sin(x)+2
    plot_train_test_data(x, y, x, y2,
                         title1="Titulo 1", xLabel1="X1", yLabel1="Y1", legend1="sin(x)",
                         title2="Titulo 2", xLabel2="X2", yLabel2="Y2", legend2="sin(x)*0.5"
                         )
    # scatter



train_test_data()
