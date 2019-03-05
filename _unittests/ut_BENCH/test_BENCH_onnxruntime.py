# -*- coding: utf-8 -*-
"""
@brief      test log(time=200s)
"""
import sys
import os
import unittest
import pandas
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase, get_temp_folder


try:
    import src
except ImportError:
    path = os.path.normpath(
        os.path.abspath(
            os.path.join(
                os.path.split(__file__)[0],
                "..",
                "..")))
    if path not in sys.path:
        sys.path.append(path)
    import src

from src.pymlbenchmark.context import machine_information
from src.pymlbenchmark.benchmark import BenchPerf
from src.pymlbenchmark.external import onnxruntime_perf_binary_classifiers


def has_onnxruntime():
    try:
        import onnxruntime
        return onnxruntime is not None
    except ImportError:
        return False


class TestBENCHonnxruntime(ExtTestCase):

    def run_onnxruntime_test(self, name, repeat=100, verbose=True):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")
        fLOG("Start '%s'" % name)
        temp = get_temp_folder(__file__, "temp_perf_onnxrt_%s" % name)

        res = onnxruntime_perf_binary_classifiers()
        sel = [r for r in res if r['name'] == name]
        if len(sel) != 1:
            raise ValueError("Unable to find one test for '%s'." % name)
        res = sel[0]

        bp = BenchPerf(res['pbefore'], res['pafter'], res['fct'])
        results = list(bp.enumerate_run_benchs(repeat=repeat, verbose=verbose))
        results_df = pandas.DataFrame(results)
        out = os.path.join(temp, "onnxruntime_%s.perf.csv" % name)
        results_df.to_csv(out, index=False)
        self.assertExists(out)

        subset = {'sklearn', 'numpy', 'pandas', 'onnxruntime',
                  'skl2onnx'}

        df = pandas.DataFrame(machine_information(subset))
        out = os.path.join(temp, "onnxruntime_%s.time.csv" % name)
        df.to_csv(out, index=False)
        self.assertExists(out)
        fLOG("Done '%s'" % name)

    @unittest.skipIf(not has_onnxruntime(), reason="onnxruntime is not installed")
    def test_bench_perf_onnxruntime_LogisticRegression(self):
        self.run_onnxruntime_test(
            self._testMethodName.split('_')[-1], verbose=False)

    @unittest.skipIf(not has_onnxruntime(), reason="onnxruntime is not installed")
    def test_bench_perf_onnxruntime_SGDClassifier(self):
        self.run_onnxruntime_test(
            self._testMethodName.split('_')[-1], verbose=False)

    @unittest.skipIf(not has_onnxruntime(), reason="onnxruntime is not installed")
    def test_bench_perf_onnxruntime_BernouilliNB(self):
        self.run_onnxruntime_test(
            self._testMethodName.split('_')[-1], verbose=False)

    @unittest.skipIf(not has_onnxruntime(), reason="onnxruntime is not installed")
    def test_bench_perf_onnxruntime_MultinomialNB(self):
        self.run_onnxruntime_test(
            self._testMethodName.split('_')[-1], verbose=False)

    @unittest.skipIf(not has_onnxruntime(), reason="onnxruntime is not installed")
    def test_bench_perf_onnxruntime_DecisionTreeClassifier(self):
        self.run_onnxruntime_test(
            self._testMethodName.split('_')[-1], verbose=False)

    @unittest.skipIf(not has_onnxruntime(), reason="onnxruntime is not installed")
    def test_bench_perf_onnxruntime_RandomForestClassifier(self):
        self.run_onnxruntime_test(
            self._testMethodName.split('_')[-1], verbose=False)


if __name__ == "__main__":
    unittest.main()
