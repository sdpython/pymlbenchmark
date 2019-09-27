"""
@file
@brief Artificial datasets.
"""
import numpy
from numpy.random import rand, randn


def random_binary_classification(N, dim):
    """
    Returns data for a binary classification problem (linear)
    with *N* observations and *dim* features.

    @param      N      number of observations
    @param      dim    number of features
    @return             *X, y*

    .. runpython::
        :showcode:

        from pymlbenchmark.datasets import random_binary_classification
        X, y = random_binary_classification(3, 6)
        print(y)
        print(X)
    """
    X_train = numpy.empty((N, dim))
    X_train[:, :] = rand(N, dim)[:, :]  # pylint: disable=E1136
    X_trainsum = X_train.sum(axis=1)
    eps = rand(N) - 0.5
    X_trainsum_ = X_trainsum + eps
    y_train = (X_trainsum_ >= X_trainsum).ravel().astype(int)
    return X_train, y_train


def random_regression(N, dim):
    """
    Returns data for a binary classification problem (linear)
    with *N* observations and *dim* features.

    @param      N      number of observations
    @param      dim    number of features
    @return             *X, y*

    .. runpython::
        :showcode:

        from pymlbenchmark.datasets import random_regression
        X, y = random_regression(3, 6)
        print(y)
        print(X)
    """
    X_train = numpy.empty((N, dim))
    X_train[:, :] = rand(N, dim)[:, :]  # pylint: disable=E1136
    eps = (randn(N, dim) - 0.5) / 4
    X_train_eps = X_train + eps
    y_train = X_train_eps.sum(
        axis=1) + numpy.power(X_train_eps / 3, 2).sum(axis=1)  # pylint: disable=E1101
    return X_train, y_train
