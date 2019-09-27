# -*- coding: utf-8 -*-
"""
@brief      test log(time=868s)
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


class TestBENCHonnxruntime_ensemble(ExtTestCase):

    def run_onnxruntime_test(self, name, repeat=2, verbose=True, stop_if_error=True):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")
        temp = get_temp_folder(__file__, "temp_perf_onnxrt_%s" % name)
        run_onnxruntime_test(temp, name, repeat=repeat, verbose=verbose,
                             N=[1, 10], dim=[1, 10],
                             stop_if_error=stop_if_error)

    @unittest.skipIf(not has_onnxruntime(), reason="onnxruntime is not installed")
    def test_bench_perf_onnxruntime_DecisionTreeClassifier(self):
        self.run_onnxruntime_test(
            self._testMethodName.split('_')[-1], verbose=False)

    @unittest.skipIf(not has_onnxruntime(), reason="onnxruntime is not installed")
    def test_bench_perf_onnxruntime_RandomForestClassifier(self):
        self.run_onnxruntime_test(
            self._testMethodName.split('_')[-1], verbose=False,
            stop_if_error=False)


if __name__ == "__main__":
    unittest.main()
