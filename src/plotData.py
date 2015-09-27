__author__ = 'tmorales'

import matplotlib.pyplot as plt


def plot_dataset(x, y, title=None, xLabel=None, yLabel=None, legend=None):
    """

    :param x:
    :param y:
    :param title:
    :param xLabel:
    :param yLabel:
    :param label:
    :return:
    """
    plt.plot(x, y, "-", linewidth=2, label=legend)
    if legend is not None: plt.legend(loc=0)
    if title is not None: plt.title(title)
    if xLabel is not None: plt.xlabel(xLabel)
    if yLabel is not None: plt.ylabel(yLabel)
    plt.show()

def plot_error(x, y, prediction, error, title=None, xLabel=None, yLabel=None, legend=None):
    """

    :param x:
    :param y:
    :param prediction:
    :param error:
    :param title:
    :param xLabel:
    :param yLabel:
    :param legend:
    :return:
    """
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
