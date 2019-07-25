import numpy as np


def mse(y, y_hat):
    error = y - y_hat
    return np.sum(error*error)/len(error)


def rmse(y, y_hat):
    return np.sqrt(mse(y, y_hat))


def mae(y, y_hat):
    error = y - y_hat
    return np.sum(np.abs(error))/len(error)
