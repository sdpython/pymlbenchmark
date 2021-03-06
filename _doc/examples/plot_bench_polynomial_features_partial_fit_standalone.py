# coding: utf-8
"""
.. _l-bench-slk-poly-standalone:

Benchmark of PolynomialFeatures + partialfit of SGDClassifier (standalone)
==========================================================================

This benchmark looks into a new implementation of
`PolynomialFeatures <https://scikit-learn.org/stable/modules/
generated/sklearn.preprocessing.PolynomialFeatures.html>`_
proposed in `PR13290 <https://github.com/
scikit-learn/scikit-learn/pull/13290>`_.
It tests the following configurations:

* **SGD**: :epkg:`sklearn:linear_model:SGDClassifier` only
* **SGD-SKL**: :epkg:`sklearn:preprocessing:PolynomialFeatures`
  from scikit-learn (no matter what it is)
* **SGD-FAST**: new implementation copy-pasted in the
  benchmark source file
* **SGD-SLOW**: implementation of 0.20.2 copy-pasted in the
  benchmark source file

This script is standalone and does not require
:epkg:`pymlbenchmark` as opposed to :ref:`l-bench-slk-poly`
which reuse functions implemented in :epkg:`pymlbenchmark`.

.. contents::
    :local:
"""
from time import perf_counter as time
import numpy
import numpy as np
from numpy.random import rand
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
# Implementations to benchmark
# ++++++++++++++++++++++++++++

def fcts_model(X, y):

    model1 = SGDClassifier()
    model2 = make_pipeline(PolynomialFeatures(), SGDClassifier())
    model3 = make_pipeline(ExtendedFeatures(kind='poly'), SGDClassifier())
    model4 = make_pipeline(ExtendedFeatures(kind='poly-slow'), SGDClassifier())

    model1.fit(PolynomialFeatures().fit_transform(X), y)
    model2.fit(X, y)
    model3.fit(X, y)
    model4.fit(X, y)

    def partial_fit_model1(X, y, model=model1):
        return model.partial_fit(X, y)

    def partial_fit_model2(X, y, model=model2):
        X2 = model.steps[0][1].transform(X)
        return model.steps[1][1].partial_fit(X2, y)

    def partial_fit_model3(X, y, model=model3):
        X2 = model.steps[0][1].transform(X)
        return model.steps[1][1].partial_fit(X2, y)

    def partial_fit_model4(X, y, model=model4):
        X2 = model.steps[0][1].transform(X)
        return model.steps[1][1].partial_fit(X2, y)

    return (partial_fit_model1, partial_fit_model2,
            partial_fit_model3, partial_fit_model4)

##############################
# Benchmarks
# ++++++++++


def build_x_y(ntrain, nfeat):
    X_train = np.empty((ntrain, nfeat))
    X_train[:, :] = rand(ntrain, nfeat)[:, :]
    X_trainsum = X_train.sum(axis=1)
    eps = rand(ntrain) - 0.5
    X_trainsum_ = X_trainsum + eps
    y_train = (X_trainsum_ >= X_trainsum).ravel().astype(int)
    return X_train, y_train


@ignore_warnings(category=(FutureWarning, DeprecationWarning))
def bench(n_obs, n_features, repeat=1000, verbose=False):
    res = []
    for n in n_obs:
        for nfeat in n_features:

            X_train, y_train = build_x_y(1000, nfeat)

            obs = dict(n_obs=n, nfeat=nfeat)

            fct1, fct2, fct3, fct4 = fcts_model(X_train, y_train)

            # creates different inputs to avoid caching in any ways
            Xs = []
            Xpolys = []
            for r in range(repeat):
                X, y = build_x_y(n, nfeat)
                Xs.append((X, y))
                Xpolys.append((PolynomialFeatures().fit_transform(X), y))

            # measure fct1
            r = len(Xs)
            st = time()
            for X, y in Xpolys:
                fct1(X, y)
            end = time()
            obs["time_sgd"] = (end - st) / r
            res.append(obs)

            # measures fct2
            st = time()
            for X, y in Xs:
                fct2(X, y)
            end = time()
            obs["time_pipe_skl"] = (end - st) / r
            res.append(obs)

            # measures fct3
            st = time()
            for X, y in Xs:
                fct3(X, y)
            end = time()
            obs["time_pipe_fast"] = (end - st) / r
            res.append(obs)

            # measures fct4
            st = time()
            for X, y in Xs:
                fct4(X, y)
            end = time()
            obs["time_pipe_slow"] = (end - st) / r
            res.append(obs)

            if verbose and (len(res) % 1 == 0 or n >= 10000):
                print("bench", len(res), ":", obs)

    return res


##############################
# Plots
# +++++

def plot_results(df, verbose=False):
    nrows = max(len(set(df.nfeat)), 2)
    ncols = max(1, 2)
    fig, ax = plt.subplots(nrows, ncols,
                           figsize=(nrows * 4, ncols * 4))
    colors = "gbry"
    row = 0
    for nfeat in sorted(set(df.nfeat)):
        pos = 0
        for _ in range(1):
            a = ax[row, pos]
            if row == ax.shape[0] - 1:
                a.set_xlabel("N observations", fontsize='x-small')
            if pos == 0:
                a.set_ylabel("Time (s) nfeat={}".format(nfeat),
                             fontsize='x-small')

            subset = df[df.nfeat == nfeat]
            if subset.shape[0] == 0:
                continue
            subset = subset.sort_values("n_obs")
            if verbose:
                print(subset)

            label = "SGD"
            subset.plot(x="n_obs", y="time_sgd", label=label, ax=a,
                        logx=True, logy=True, c=colors[0], style='--')
            label = "SGD-SKL"
            subset.plot(x="n_obs", y="time_pipe_skl", label=label, ax=a,
                        logx=True, logy=True, c=colors[1], style='--')
            label = "SGD-FAST"
            subset.plot(x="n_obs", y="time_pipe_fast", label=label, ax=a,
                        logx=True, logy=True, c=colors[2])
            label = "SGD-SLOW"
            subset.plot(x="n_obs", y="time_pipe_slow", label=label, ax=a,
                        logx=True, logy=True, c=colors[3])

            a.legend(loc=0, fontsize='x-small')
            if row == 0:
                a.set_title("--", fontsize='x-small')
            pos += 1
        row += 1

    plt.suptitle("Benchmark for Polynomial with SGDClassifier", fontsize=16)


#################################
# Final function for the benchmark
# ++++++++++++++++++++++++++++++++

def run_bench(repeat=100, verbose=False):
    n_obs = [10, 100, 1000]
    n_features = [5, 10, 50]

    with sklearn.config_context(assume_finite=True):
        start = time()
        results = bench(n_obs, n_features, repeat=repeat, verbose=verbose)
        end = time()

    results_df = pandas.DataFrame(results)
    print("Total time = %0.3f sec\n" % (end - start))

    # plot the results
    plot_results(results_df, verbose=verbose)
    return results_df

#################################
# Run the benchmark
# +++++++++++++++++


print("numpy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
df = run_bench(verbose=True)
print(df)

plt.show()
