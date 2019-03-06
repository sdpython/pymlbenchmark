# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""

import sys
import os
import unittest
from pyquickhelper.pycode import ExtTestCase


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


from src.pymlbenchmark.datasets import experiment_results


class TestRealData(ExtTestCase):

    def test_experiment_results(self):
        res = experiment_results('onnxruntime_LogisticRegression')
        self.assertEqual(res.shape, (112, 12))


if __name__ == "__main__":
    unittest.main()
