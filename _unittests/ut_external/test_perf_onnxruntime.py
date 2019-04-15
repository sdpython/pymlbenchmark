# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import os
import unittest
import pandas
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from pyquickhelper.texthelper import compare_module_version
from pymlbenchmark.context import machine_information
from pymlbenchmark.benchmark import BenchPerf
from pymlbenchmark.external import onnxruntime_perf_binary_classifiers


def has_onnxruntime(version):
    try:
        import onnxruntime
        return compare_module_version(onnxruntime.__version__, version) >= 0
    except ImportError:
        return None


class TestPerfOnnxRuntime(ExtTestCase):

    def test_bench_list(self):
        res = onnxruntime_perf_binary_classifiers()
        self.assertGreater(len(res), 1)

    @unittest.skipIf(not has_onnxruntime('0.3.0'),
                     reason="onnxruntime is not installed")
    def test_perf_onnxruntime_logreg(self):
        res = onnxruntime_perf_binary_classifiers()[0]

        bp = BenchPerf(res['pbefore'], res['pafter'], res['fct'])
        results = list(bp.enumerate_run_benchs(repeat=10, verbose=True))
        results_df = pandas.DataFrame(results)
        temp = get_temp_folder(__file__, "temp_perf_onnxruntime_logreg")
        out = os.path.join(temp, "onnxruntime_logreg.perf.csv")
        results_df.to_csv(out, index=False)
        self.assertExists(out)

        subset = {'sklearn', 'numpy', 'pandas', 'onnxruntime',
                  'skl2onnx'}

        df = pandas.DataFrame(machine_information(subset))
        out = os.path.join(temp, "onnxruntime_logreg.time.csv")
        df.to_csv(out, index=False)
        self.assertExists(out)


if __name__ == "__main__":
    unittest.main()
