# coding: utf-8
"""
.. _l-example-onnxruntime-logreg:

Benchmark of onnxruntime on DecisionTree
========================================

The example uses what :epkg:`pymlbenchmark` implements.
"""
import pandas
from time import perf_counter as time
from sklearn.linear_model import LogisticRegression
from pymlbenchmark.context import machine_information
from pymlbenchmark.benchmark import BenchPerf
from pymlbenchmark.external import OnnxRuntimeBenchPerfTestBinaryClassification


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


if __name__ == '__main__':
    df = run_bench(verbose=True)
    df.to_csv("bench_plot_onnxruntime_logistic_regression.perf.csv", index=False)
    print(df.head())

    pkgs = ['numpy', 'pandas', 'sklearn', 'skl2onnx', 'onnxruntime', 'onnx']
    df = pandas.DataFrame(machine_information(pkgs))
    df.to_csv("bench_plot_onnxruntime_logistic_regression.time.csv", index=False)
    print(df)
