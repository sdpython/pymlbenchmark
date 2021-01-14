# -*- coding: utf-8 -*-
"""
@brief      test log(time=10s)
"""
import unittest
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from pymlbenchmark.external import run_onnxruntime_test


def has_onnxruntime():
    try:
        import onnxruntime  # pylint: disable=C0415
        return onnxruntime is not None
    except ImportError:
        return False


class TestBENCHonnxruntime_linear(ExtTestCase):

    def run_onnxruntime_test(self, name, repeat=100, verbose=True, stop_if_error=True):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")
        temp = get_temp_folder(__file__, "temp_perf_onnxrt_%s" % name)
        run_onnxruntime_test(temp, name, repeat=repeat, verbose=verbose,
                             stop_if_error=stop_if_error, N_fit=1000,
                             dim=[1, 5])

    def test_bench_perf_onnxruntime_LogisticRegression(self):
        self.run_onnxruntime_test(
            self._testMethodName.split('_')[-1], verbose=False)

    def test_bench_perf_onnxruntime_SGDClassifier(self):
        self.run_onnxruntime_test(
            self._testMethodName.split('_')[-1], verbose=False)


if __name__ == "__main__":
    unittest.main()
