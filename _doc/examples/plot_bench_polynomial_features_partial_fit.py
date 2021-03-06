# coding: utf-8
"""
.. _l-bench-slk-poly:

Benchmark of PolynomialFeatures + partialfit of SGDClassifier
=============================================================

This benchmark looks into a new implementation of
`PolynomialFeatures <https://scikit-learn.org/stable/
modules/generated/sklearn.preprocessing.PolynomialFeatures.html>`_
proposed in `PR13290 <https://github.com/
scikit-learn/scikit-learn/pull/13290>`_.
It tests the following configurations:

* **SGD-ONLY**: :epkg:`sklearn:linear_model:SGDClassifier` only
* **SGD-SKL**: :epkg:`sklearn:preprocessing:PolynomialFeature`
  from :epkg:`scikit-learn` (no matter what it is)
* **SGD-FAST**: new implementation copy-pasted in the
  benchmark source file
* **SGD-SLOW**: implementation of 0.20.2 copy-pasted
  in the benchmark source file

This example takes the example :ref:`l-bench-slk-poly-standalone`
and rewrites it with module :epkg:`pymlbenchmark`.

.. contents::
    :local:
"""
from pymlbenchmark.plotting import plot_bench_results
from pymlbenchmark.context import machine_information
from time import perf_counter as time

import matplotlib.pyplot as plt
import pandas
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDClassifier
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from mlinsights.mlmodel import ExtendedFeatures


##############################
# Implementation to benchmark
# +++++++++++++++++++++++++++

from pymlbenchmark.benchmark import BenchPerf, BenchPerfTest
from pymlbenchmark.datasets import random_binary_classification


class PolyBenchPerfTest(BenchPerfTest):
    def __init__(self, dim=None, **opts):
        # Models are fitted here. Every not measured
        # should take place here.
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
        # The benchmark requires a new datasets each time.
        assert N is not None
        assert dim is not None
        return random_binary_classification(N, dim)

    def fcts(self, dim=None, **kwargs):
        # The function returns the prediction functions to tests.

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

    def validate(self, results, **kwargs):
        for ind, row, model in results:
            assert isinstance(row, dict)  # test options
            assert isinstance(model, SGDClassifier)  # trained model


##############################
# Benchmark function
# ++++++++++++++++++

@ignore_warnings(category=(FutureWarning, DeprecationWarning))
def run_bench(repeat=100, verbose=False):
    pbefore = dict(dim=[5, 10, 50])
    pafter = dict(N=[10, 100, 1000])
    bp = BenchPerf(pbefore, pafter, PolyBenchPerfTest)

    with sklearn.config_context(assume_finite=True):
        start = time()
        results = list(bp.enumerate_run_benchs(repeat=repeat, verbose=verbose))
        end = time()

    results_df = pandas.DataFrame(results)
    print("Total time = %0.3f sec\n" % (end - start))
    return results_df

##############################
# Run the benchmark
# +++++++++++++++++


df = run_bench(verbose=True)
df.to_csv("plot_bench_polynomial_features_partial_fit.perf.csv", index=False)
print(df.head())

#########################
# Extract information about the machine used
# ++++++++++++++++++++++++++++++++++++++++++

pkgs = ['numpy', 'pandas', 'sklearn']
dfi = pandas.DataFrame(machine_information(pkgs))
dfi.to_csv("plot_bench_polynomial_features_partial_fit.time.csv", index=False)
print(dfi)

#############################
# Plot the results
# ++++++++++++++++

print(df.columns)
plot_bench_results(df, row_cols='N', col_cols=None,
                   x_value='dim', hue_cols=None,
                   cmp_col_values='test',
                   title="PolynomialFeatures + partial_fit\n"
                   "Benchmark scikit-learn PR13290")
plt.show()
