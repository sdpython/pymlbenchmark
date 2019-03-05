# -*- coding: utf-8 -*-
"""
@brief      test log(time=200s)
"""
import sys
import os
import unittest
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

from src.pymlbenchmark.external import run_onnxruntime_test


def has_onnxruntime():
    try:
        import onnxruntime
        return onnxruntime is not None
    except ImportError:
        return False


class TestBENCHonnxruntime_ensemble(ExtTestCase):

    def run_onnxruntime_test(self, name, repeat=100, verbose=True, stop_if_error=True):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")
        temp = get_temp_folder(__file__, "temp_perf_onnxrt_%s" % name)
        run_onnxruntime_test(temp, name, repeat=repeat, verbose=verbose,
                             stop_if_error=stop_if_error)

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
