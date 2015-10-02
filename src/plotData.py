__author__ = 'tmorales'

import matplotlib.pyplot as plt


def plot_twoHorizontalPlots(x1, y1, x2, y2,
                            type1="line", title1=None, xLabel1=None, yLabel1=None, linestyle1=None, legend1=None,
                            type2="line", title2=None, xLabel2=None, yLabel2=None, linestyle2=None, legend2=None):
    """Two horizontal plots with one dataset for each one"""
    f, ax = plt.subplots(2, 1, figsize=((15,10)))
    # plot 1
    if type1 is "line":
        ax[0].plot(x1, y1, linestyle=linestyle1, linewidth=1, label=legend1)
    if type1 is "scatter":
        pass
    if title1 is not None: ax[0].set_title(title1)
    if xLabel1 is not None: ax[0].set_xlabel(xLabel1)
    if yLabel1 is not None: ax[0].set_ylabel(yLabel1)
    ax[0].legend(loc=0)

    # plot 2
    if type2 is "line":
        ax[1].plot(x2, y2, linestyle=linestyle2, linewidth=2, label=legend2)
    if type2 is "scatter":
        pass
    if title2 is not None: ax[1].set_title(title2)
    if xLabel2 is not None: ax[1].set_xlabel(xLabel2)
    if yLabel2 is not None: ax[1].set_ylabel(yLabel2)
    ax[1].legend(loc=0)
    plt.show()

def plot_dataset(x, y, title=None, xLabel=None, yLabel=None, legend=None):
    """One plot without kind"""
    plt.plot(x, y, "-", linewidth=2, label=legend)
    if legend is not None: plt.legend(loc=0)
    if title is not None: plt.title(title)
    if xLabel is not None: plt.xlabel(xLabel)
    if yLabel is not None: plt.ylabel(yLabel)
    plt.show()

def plot_error(x, y, prediction, error, title=None, xLabel=None, yLabel=None, legend=None):
    """two vertical plots, the lower one has two datasets"""
    f, ax = plt.subplots(2, 1, figsize=((15,8)))
    # plot 1
    ax[0].plot(range(len(error)), error, linewidth=2)
    if title is not None: ax[0].set_title(title)
    if xLabel is not None: ax[0].set_xlabel(xLabel)
    if yLabel is not None: ax[0].set_ylabel(yLabel)

    # plot 2
    ax[1].plot(x, y, "-.", linewidth=2, label="train get")
    ax[1].plot(x, prediction, "-", linewidth=2, label="net output")
    ax[1].legend(loc=0)
    plt.show()


def plot_train_test_data(x_train, y_train, x_test, y_test,
                         title1=None, xLabel1=None, yLabel1=None, legend1=None,
                         title2=None, xLabel2=None, yLabel2=None, legend2=None):
    """plot the train and test data in two horizontal plots"""
    plot_twoHorizontalPlots(x_train, y_train, x_test, y_test,
                            title1=title1, xLabel1=xLabel1, yLabel1=yLabel1, linestyle1="-", legend1=legend1,
                            title2=title2, xLabel2=xLabel2, yLabel2=yLabel2, linestyle2="-", legend2=legend2)