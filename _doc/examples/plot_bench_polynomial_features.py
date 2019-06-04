# coding: utf-8
"""
Benchmark of PolynomialFeatures in scikit-learn 0.20.0
======================================================

This benchmark looks into a new implementation of
`PolynomialFeatures <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html>`_
proposed in `PR13290 <https://github.com/scikit-learn/scikit-learn/pull/13290>`_.

.. contents::
    :local:
"""
from time import time
from itertools import combinations, chain
from itertools import combinations_with_replacement as combinations_w_r

import numpy as np
from numpy.random import rand
from numpy.testing import assert_almost_equal
import matplotlib.pyplot as plt
import pandas

from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.testing import ignore_warnings


##############################
# Implementations to benchmark
# ++++++++++++++++++++++++++++

from pymlbenchmark.benchmark import BenchPerf, BenchPerfTest
from pymlbenchmark.datasets import random_binary_classification


class PolyBenchPerfTest(BenchPerfTest):

    def __init__(self, dim=None, degree=2, interaction_only=False,
                 order='C', **opts):
        # Models are fitted here. Every instruction not measured
        # should take place here.
        assert dim is not None
        BenchPerfTest.__init__(self, **opts)

        self.order = order

        # PolynomialFeature 0.20.2 for dense matrices (no bias).
        comb = self._combinations(dim, degree, interaction_only,
                                  include_bias=False)
        self.comb = list(comb)

        # Current PolynomialFeatures.
        try:
            model = PolynomialFeatures(degree=degree, include_bias=False,
                                       interaction_only=interaction_only,
                                       order=order)
        except TypeError:
            # order parameter added in 0.21
            model = PolynomialFeatures(degree=degree, include_bias=False,
                                       interaction_only=interaction_only)
        X = self.data(1, dim)[0]
        self.model = model.fit(X)

    def _combinations(self, n_features, degree, interaction_only, include_bias):
        comb = (combinations if interaction_only else combinations_w_r)
        start = int(not include_bias)
        return chain.from_iterable(comb(range(n_features), i)
                                   for i in range(start, degree + 1))

    def data(self, N=None, dim=None, **opts):
        # The benchmark requires a new datasets each time.
        assert N is not None
        assert dim is not None
        return random_binary_classification(N, dim)[:1]

    def fcts(self, dim=None, **kwargs):
        # The function returns the prediction functions to tests.

        def fct_polynomial_features_0_20_2(X):
            XP = np.empty((X.shape[0], len(self.comb)),
                          dtype=X.dtype, order=self.order)
            for i, comb in enumerate(self.comb):
                XP[:, i] = X[:, comb].prod(1)
            return XP

        def compute_feat_dev(X):
            return self.model.transform(X)

        return [{'test': 'PF-0.20.2', 'fct': fct_polynomial_features_0_20_2},
                {'test': 'PF-DEV', 'fct': compute_feat_dev}]

    def validate(self, results, **kwargs):
        """
        Checks that methods *predict* and *predict_proba* returns
        the same results for both :epkg:`scikit-learn` and
        :epkg:`onnxruntime`.
        """
        res = {}
        baseline = None
        for idt, fct, vals in results:
            key = idt, fct.get('method', '')
            if key not in res:
                res[key] = {}
            if isinstance(vals, list):
                vals = pandas.DataFrame(vals).values
            lib = fct['test']
            res[key][lib] = vals
            if lib == 'PF-0.20.2':
                baseline = lib

        if len(res) == 0:
            raise RuntimeError("No results to compare.")
        if baseline is None:
            raise RuntimeError(
                "Unable to guess the baseline in {}.".format(list(res.pop())))

        for key, exp in res.items():
            vbase = exp[baseline]
            if vbase.shape[0] <= 10000:
                for name, vals in exp.items():
                    if name == baseline:
                        continue
                    p1, p2 = vbase, vals
                    if len(p1.shape) == 1 and len(p2.shape) == 2:
                        p2 = p2.ravel()
                    try:
                        assert_almost_equal(p1, p2, decimal=4)
                    except AssertionError as e:
                        msg = "ERROR: Dim {} - discrepencies between\n{} and\n{} for {}.".format(
                            p1.shape, baseline, name, key)
                        self.dump_error(msg, skl=self.skl, ort=self.ort,
                                        results=results, **kwargs)
                        raise AssertionError(msg) from e


##############################
# Benchmark
# +++++++++


def allow_configuration(N=None, dim=None, degree=None,
                        interaction_only=None, order=None):
    if dim is not None and dim >= 60 and \
            degree is not None and degree >= 4 and \
            N is not None and N < 100:
        return False
    if dim is not None and dim >= 60 and \
            degree is not None and degree >= 3 and \
            N is not None and N >= 100:
        return False
    if dim is not None and dim >= 40 and \
            degree is not None and degree >= 4:
        return False
    if N is not None and N >= 10000 and \
            degree is not None and degree >= 4:
        return False
    if N is not None and N >= 100000 and \
            degree is not None and degree >= 3 and \
            dim is not None and dim >= 40:
        return False
    return True


@ignore_warnings(category=FutureWarning)
def run_bench(repeat=10, verbose=False):
    pbefore = dict(dim=[2, 5, 10, 20, 50], degree=[2, 3],
                   interaction_only=[False, True])
    pafter = dict(N=[1, 10, 100, 1000], order=['C', 'F'])

    bp = BenchPerf(pbefore, pafter, PolyBenchPerfTest,
                   filter_test=allow_configuration)

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
df.to_csv("plot_bench_polynomial_features.perf.csv", index=False)
print(df.head())

#########################
# Extract information about the machine used
# ++++++++++++++++++++++++++++++++++++++++++

from pymlbenchmark.context import machine_information
pkgs = ['numpy', 'pandas', 'sklearn']
dfi = pandas.DataFrame(machine_information(pkgs))
dfi.to_csv("plot_bench_polynomial_features.time.csv", index=False)
print(dfi)

#############################
# Plot the results
# ++++++++++++++++

from pymlbenchmark.plotting import plot_bench_results
print(df.columns)
plot_bench_results(df, row_cols=['N', 'order'],
                   col_cols=['degree'], x_value='dim',
                   hue_cols=['interaction_only'],
                   cmp_col_values='test',
                   title="PolynomialFeatures\nBenchmark scikit-learn PR13290")
plt.show()
