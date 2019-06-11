# coding: utf-8
"""
Investigate a failure from a benchmark
======================================

The method ``validate`` may raise an exception and
in that case, the class :class:`BenchPerfTest
<pymlbenchmark.benchmark.benchmark_perf.BenchPerfTest>`.
The following script shows how to investigate.

.. contents::
    :local:
"""
from time import time
import numpy
from numpy.random import rand
from numpy.testing import assert_almost_equal
import matplotlib.pyplot as plt
import pandas
from scipy.special import expit
from sklearn.utils.testing import ignore_warnings
from sklearn.linear_model import LogisticRegression
from pymlbenchmark.benchmark import BenchPerf
from pymlbenchmark.external import OnnxRuntimeBenchPerfTestBinaryClassification


##############################
# Defines the benchmark and runs it
# +++++++++++++++++++++++++++++++++


class OnnxRuntimeBenchPerfTestBinaryClassification3(OnnxRuntimeBenchPerfTestBinaryClassification):
    """
    Overwrites the class to add a pure python implementation
    of the logistic regression.
    """

    def fcts(self, dim=None, **kwargs):

        def predict_py_predict(X, model=self.skl):
            coef = model.coef_
            intercept = model.intercept_
            pred = numpy.dot(X, coef.T) + intercept
            return (pred >= 0).astype(numpy.int32)

        def predict_py_predict_proba(X, model=self.skl):
            coef = model.coef_
            intercept = model.intercept_
            pred = numpy.dot(X, coef.T) + intercept
            decision_2d = numpy.c_[-pred, pred]
            return expit(decision_2d)

        res = OnnxRuntimeBenchPerfTestBinaryClassification.fcts(
            self, dim=dim, **kwargs)
        res.extend([
            {'method': 'predict', 'lib': 'py', 'fct': predict_py_predict},
            {'method': 'predict_proba', 'lib': 'py',
                'fct': predict_py_predict_proba},
        ])
        return res

    def validate(self, results, **kwargs):
        """
        Raises an exception and locally dump everything we need
        to investigate.
        """
        # Checks that methods *predict* and *predict_proba* returns
        # the same results for both scikit-learn and onnxruntime.
        OnnxRuntimeBenchPerfTestBinaryClassification.validate(
            self, results, **kwargs)

        # Let's dump anything we need for later.
        # kwargs contains the input data.
        self.dump_error("Just for fun", skl=self.skl,
                        ort_onnx=self.ort_onnx,
                        results=results, **kwargs)
        raise AssertionError("Just for fun")


@ignore_warnings(category=FutureWarning)
def run_bench(repeat=10, verbose=False):

    pbefore = dict(dim=[1, 5], fit_intercept=[True])
    pafter = dict(N=[1, 10, 100])
    test = lambda dim=None, **opts: OnnxRuntimeBenchPerfTestBinaryClassification3(
        LogisticRegression, dim=dim, **opts)
    bp = BenchPerf(pbefore, pafter, test)

    start = time()
    results = list(bp.enumerate_run_benchs(repeat=repeat, verbose=verbose))
    end = time()

    results_df = pandas.DataFrame(results)
    print("Total time = %0.3f sec\n" % (end - start))
    return results_df


########################
# Runs the benchmark.
try:
    run_bench(verbose=True)
except AssertionError as e:
    print(e)

#############################
# Investigation
# +++++++++++++
#
# Let's retrieve what was dumped.

from pickle import load
from onnxruntime import InferenceSession
filename = "BENCH-ERROR-OnnxRuntimeBenchPerfTestBinaryClassification3-0.pkl"
with open(filename, "rb") as f:
    data = load(f)

print(list(sorted(data)))
print("msg:", data["msg"])
print(list(sorted(data["data"])))
print(data["data"]['skl'])

##################################
# The input data is the following:

print(data['data']['data'])

########################################
# Let's compare predictions.

model_skl = data["data"]['skl']
model_onnx = InferenceSession(data["data"]['ort_onnx'].SerializeToString())
input_name = model_onnx.get_inputs()[0].name


def ort_predict_proba(sess, input, input_name):
    res = model_onnx.run(None, {input_name: input.astype(numpy.float32)})[1]
    return pandas.DataFrame(res).values


pred_skl = [model_skl.predict_proba(input[0])
            for input in data['data']['data']]
pred_onnx = [ort_predict_proba(model_onnx, input[0], input_name)
             for input in data['data']['data']]

print(pred_skl)
print(pred_onnx)

##############################
# They look the same. Let's check...

for a, b in zip(pred_skl, pred_onnx):
    assert_almost_equal(a, b)

###################################
# Computing differences.


def diff(a, b):
    return numpy.max(numpy.abs(a.ravel() - b.ravel()))


diffs = list(sorted(diff(a, b) for a, b in zip(pred_skl, pred_onnx)))

import matplotlib.pyplot as plt
plt.plot(diffs)
plt.title("Differences between prediction with\nscikit-learn and onnxruntime"
          "\nfor Logistic Regression")
plt.show()
