# coding: utf-8
"""
.. _l-example-onnxruntime-logreg:

Benchmark of onnxruntime on DecisionTree
========================================

The example uses what :epkg:`pymlbenchmark` implements.
"""
from time import perf_counter as time
import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.utils.testing import ignore_warnings
from pymlbenchmark.context import machine_information
from pymlbenchmark.benchmark import BenchPerf
from pymlbenchmark.external import OnnxRuntimeBenchPerfTestBinaryClassification
from pymlbenchmark.plotting import plot_bench_results


@ignore_warnings(category=FutureWarning)
def run_bench(repeat=100, verbose=False):

    pbefore = dict(dim=[1, 5, 10, 20, 50, 100, 150],
                   fit_intercept=[True, False])
    pafter = dict(N=[1, 10])
    test = lambda dim=None, **opts: OnnxRuntimeBenchPerfTestBinaryClassification(
        LogisticRegression, dim=dim, **opts)
    bp = BenchPerf(pbefore, pafter, test)

    start = time()
    results = list(bp.enumerate_run_benchs(repeat=repeat, verbose=verbose))
    end = time()

    results_df = pandas.DataFrame(results)
    print("Total time = %0.3f sec\n" % (end - start))
    return results_df

###################
# Runs the benchmark.


df = run_bench(verbose=True)
df.to_csv("bench_plot_onnxruntime_logistic_regression.perf.csv", index=False)
print(df.head())

#########################
# Extract information about the machine used.
pkgs = ['numpy', 'pandas', 'sklearn', 'skl2onnx', 'onnxruntime', 'onnx']
dfi = pandas.DataFrame(machine_information(pkgs))
dfi.to_csv("bench_plot_onnxruntime_logistic_regression.time.csv", index=False)
print(dfi)

#############################
# Plots the results.
plot_bench_results(df, row_cols=['N'], col_cols=['method'],
                   x_value='dim', hue_cols=['fit_intercept'],
                   title="LogisticRegression\nBenchmark scikit-learn / onnxruntime")
plt.show()
