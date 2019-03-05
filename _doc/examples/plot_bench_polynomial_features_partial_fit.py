# coding: utf-8
"""
.. _l-bench-slk-poly:

Benchmark of PolynomialFeatures
===============================

This benchmark looks into a new implementation of
`PolynomialFeatures <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html>`_
proposed in `PR13290 <https://github.com/scikit-learn/scikit-learn/pull/13290>`_.
It tests the following configurations:

* *SGD*: *SGDClassifier* only
* *SGD-SKL*: *PolynomialFeatures* from scikit-learn (no matter what it is)
* *SGD-FAST*: new implementation copy-pasted in the benchmark source file
* *SGD-SLOW*: implementation of 0.20.2 copy-pasted in the benchmark source file

This example takes the example :ref:`l-bench-slk-poly-standalone`
and rewrites it with module :epkg:`pymlbenchmark`.

.. contents::
    :local:
"""
from io import BytesIO
from time import perf_counter as time
from itertools import combinations, chain
from itertools import combinations_with_replacement as combinations_w_r
import io
import os
import sys

import numpy as np
from numpy.random import rand
from numpy.testing import assert_almost_equal
import matplotlib
import matplotlib.pyplot as plt
import pandas
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDClassifier
from sklearn.utils.testing import ignore_warnings
from mlinsights.mlmodel import ExtendedFeatures


##############################
# Implementations to benchmark
# ++++++++++++++++++++++++++++

from pymlbenchmark.benchmark import BenchPerf, BenchPerfTest
from pymlbenchmark.datasets import random_binary_classification


class PolyBenchPerfTest(BenchPerfTest):
    def __init__(self, dim=None, **opts):
        assert dim is not None
        BenchPerfTest.__init__(self, **opts)
        self.model1 = SGDClassifier()
        self.model2 = make_pipeline(PolynomialFeatures(), SGDClassifier())
        self.model3 = make_pipeline(
            ExtendedFeatures(kind='poly'), SGDClassifier())
        self.model4 = make_pipeline(ExtendedFeatures(
            kind='poly-slow'), SGDClassifier())
        X, y = random_binary_classification(10000, dim)
        self.model1.fit(PolynomialFeatures().fit_transform(X), y)
        self.model2.fit(X, y)
        self.model3.fit(X, y)
        self.model4.fit(X, y)

    def data(self, N=None, dim=None):
        assert N is not None
        assert dim is not None
        return random_binary_classification(N, dim)

    def fcts(self, dim=None, **kwargs):
        def preprocess(X, y):
            return PolynomialFeatures().fit_transform(X), y

        def partial_fit_model1(X, y, model=self.model1):
            return model.partial_fit(X, y)

        def partial_fit_model2(X, y, model=self.model2):
            X2 = model.steps[0][1].transform(X)
            return model.steps[1][1].partial_fit(X2, y)

        def partial_fit_model3(X, y, model=self.model3):
            X2 = model.steps[0][1].transform(X)
            return model.steps[1][1].partial_fit(X2, y)

        def partial_fit_model4(X, y, model=self.model4):
            X2 = model.steps[0][1].transform(X)
            return model.steps[1][1].partial_fit(X2, y)

        return [{'test': 'SGD-ONLY', 'fct': (preprocess, partial_fit_model1)},
                {'test': 'SGD-SKL', 'fct': partial_fit_model2},
                {'test': 'SGD-FAST', 'fct': partial_fit_model3},
                {'test': 'SGD-SLOW', 'fct': partial_fit_model4}]

    def validate(self, results):
        for row in results:
            assert isinstance(row[0], dict)  # test options
            assert isinstance(row[1], SGDClassifier)  # trained model


##############################
# Benchmarks
# ++++++++++


@ignore_warnings(category=FutureWarning)
def run_bench(repeat=100, verbose=False):
    pbefore = dict(dim=[5, 10, 50])
    pafter = dict(N=[10, 100, 1000])
    bp = BenchPerf(pbefore, pafter, PolyBenchPerfTest)

    start = time()
    results = list(bp.enumerate_run_benchs(repeat=repeat, verbose=verbose))
    end = time()

    results_df = pandas.DataFrame(results)
    print("Total time = %0.3f sec\n" % (end - start))
    return results_df


import sklearn
import numpy
print("numpy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
df = run_bench(verbose=True)
print(df)
df.to_csv('results.csv', index=False)

# plt.show()
