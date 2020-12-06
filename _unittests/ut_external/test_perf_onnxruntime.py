# -*- coding: utf-8 -*-
"""
@brief      test log(time=14s)
"""
import os
import unittest
import pandas
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from pyquickhelper.texthelper import compare_module_version
from pymlbenchmark.context import machine_information
from pymlbenchmark.benchmark import BenchPerf
from pymlbenchmark.external import (
    onnxruntime_perf_binary_classifiers,
    onnxruntime_perf_regressors,
    OnnxRuntimeBenchPerfTestBinaryClassification)


def has_onnxruntime(version):
    try:
        import onnxruntime  # pylint: disable=C0415
        return compare_module_version(onnxruntime.__version__, version) >= 0
    except ImportError:
        return None


class MyBenchTest(OnnxRuntimeBenchPerfTestBinaryClassification):

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
            lib = fct['lib']
            res[key][lib] = vals
            if lib == 'skl':
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
                    msg = "ERROR: Dim {}-{} - discrepencies between '{}' and '{}' for '{}'.".format(
                        vbase.shape, vals.shape, baseline, name, key)
                    self.dump_error(msg, skl=self.skl, ort=self.ort,
                                    baseline=vbase, discrepencies=vals,
                                    onnx_bytes=self.ort_onnx.SerializeToString(),
                                    results=results, **kwargs)
                    raise AssertionError(
                        "Always different for unit tests purposes.")


class TestPerfOnnxRuntime(ExtTestCase):

    def test_bench_list(self):
        res = onnxruntime_perf_binary_classifiers()
        self.assertGreater(len(res), 1)

    def test_perf_onnxruntime_logreg_fails(self):
        res = onnxruntime_perf_binary_classifiers(MyBenchTest)[0]

        bp = BenchPerf(res['pbefore'], res['pafter'], res['fct'])
        results = list(bp.enumerate_run_benchs(
            repeat=10, verbose=True, stop_if_error=False))
        results_df = pandas.DataFrame(results)
        su = results_df['error_c'].sum()
        self.assertEqual(su, results_df.shape[0])
        temp = get_temp_folder(__file__, "temp_perf_onnxruntime_logreg_fails")
        out = os.path.join(temp, "onnxruntime_logreg.perf.csv")
        results_df.to_csv(out, index=False)
        self.assertExists(out)

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
                  'skl2onnx', 'mlprodict'}

        df = pandas.DataFrame(machine_information(subset))
        out = os.path.join(temp, "onnxruntime_logreg.time.csv")
        df.to_csv(out, index=False)
        self.assertExists(out)

    def test_perf_onnxruntime_linreg(self):
        res = onnxruntime_perf_regressors()[0]

        bp = BenchPerf(res['pbefore'], res['pafter'], res['fct'])
        results = list(bp.enumerate_run_benchs(repeat=10, verbose=True))
        results_df = pandas.DataFrame(results)
        temp = get_temp_folder(__file__, "temp_perf_onnxruntime_linreg")
        out = os.path.join(temp, "onnxruntime_linreg.perf.csv")
        results_df.to_csv(out, index=False)
        self.assertExists(out)

        subset = {'sklearn', 'numpy', 'pandas', 'onnxruntime',
                  'skl2onnx', 'mlprodict'}

        df = pandas.DataFrame(machine_information(subset))
        out = os.path.join(temp, "onnxruntime_linreg.time.csv")
        df.to_csv(out, index=False)
        self.assertExists(out)

    def test_perf_onnxruntime_gpr64(self):
        res = onnxruntime_perf_regressors()[3]

        bp = BenchPerf(res['pbefore'], res['pafter'], res['fct'])
        results = list(bp.enumerate_run_benchs(
            repeat=10, verbose=True, stop_if_error=False))
        results_df = pandas.DataFrame(results)
        temp = get_temp_folder(__file__, "temp_perf_onnxruntime_gpr")
        out = os.path.join(temp, "onnxruntime_gpr.perf.csv")
        results_df.to_csv(out, index=False)
        self.assertExists(out)

        subset = {'sklearn', 'numpy', 'pandas', 'onnxruntime',
                  'skl2onnx', 'mlprodict'}

        df = pandas.DataFrame(machine_information(subset))
        out = os.path.join(temp, "onnxruntime_linreg.time.csv")
        df.to_csv(out, index=False)
        self.assertExists(out)


if __name__ == "__main__":
    unittest.main()
