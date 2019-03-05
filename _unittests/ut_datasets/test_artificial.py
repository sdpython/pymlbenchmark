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


from src.pymlbenchmark.datasets import random_binary_classification


class TestArtificial(ExtTestCase):

    def test_random_binary_classification(self):
        X, y = random_binary_classification(40, 4)
        self.assertEqual(X.shape, (40, 4))
        self.assertEqual(y.shape, (40, ))
        self.assertEqual(len(set(y)), 2)


if __name__ == "__main__":
    unittest.main()
