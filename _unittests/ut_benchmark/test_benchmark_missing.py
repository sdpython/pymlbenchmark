# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from pymlbenchmark.benchmark import BenchPerfTest


class TestBenchPerMissing(ExtTestCase):

    def test_bench_not(self):
        bp = BenchPerfTest()
        self.assertRaise(lambda: bp.data(), NotImplementedError)
        self.assertRaise(lambda: bp.fcts(), NotImplementedError)


if __name__ == "__main__":
    unittest.main()
