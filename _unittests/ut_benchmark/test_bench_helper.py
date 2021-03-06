# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import os
import unittest
import pandas
from pyquickhelper.pycode import ExtTestCase
from pymlbenchmark.benchmark.bench_helper import bench_pivot


class TestBenchHelper(ExtTestCase):

    def test_bench_pivot(self):
        data = os.path.join(os.path.dirname(__file__), "data",
                            "plot_bench_polynomial_features.perf.csv")
        df = pandas.read_csv(data)
        bench = bench_pivot(df, experiment='test')
        self.assertEqual(list(bench.columns), ['PF-0.20.2', 'PF-DEV'])
        self.assertEqual(bench.shape, (64, 2))


if __name__ == "__main__":
    unittest.main()
