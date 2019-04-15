# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from pymlbenchmark.datasets import experiment_results


class TestRealData(ExtTestCase):

    def test_experiment_results(self):
        res = experiment_results('onnxruntime_LogisticRegression')
        self.assertEqual(res.shape, (112, 15))


if __name__ == "__main__":
    unittest.main()
