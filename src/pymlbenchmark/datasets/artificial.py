"""
@file
@brief Artificial datasets.
"""
import numpy
from numpy.random import rand


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
