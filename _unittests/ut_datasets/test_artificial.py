# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from pymlbenchmark.datasets import random_binary_classification


class TestArtificial(ExtTestCase):

    def test_random_binary_classification(self):
        X, y = random_binary_classification(40, 4)
        self.assertEqual(X.shape, (40, 4))
        self.assertEqual(y.shape, (40, ))
        self.assertEqual(len(set(y)), 2)


if __name__ == "__main__":
    unittest.main()
