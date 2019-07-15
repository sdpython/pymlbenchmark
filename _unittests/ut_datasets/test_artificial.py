# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from pymlbenchmark.datasets import random_binary_classification, random_regression


class TestArtificial(ExtTestCase):

    def test_random_binary_classification(self):
        X, y = random_binary_classification(40, 4)
        self.assertEqual(X.shape, (40, 4))
        self.assertEqual(y.shape, (40, ))
        self.assertEqual(len(set(y)), 2)
        self.assertEqual(y.dtype, numpy.int32)

    def test_random_regression(self):
        X, y = random_regression(40, 4)
        self.assertEqual(X.shape, (40, 4))
        self.assertEqual(y.shape, (40, ))
        self.assertEqual(y.dtype, X.dtype)


if __name__ == "__main__":
    unittest.main()
