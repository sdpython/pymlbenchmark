# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
from sklearn.ensemble import RandomForestClassifier
from pyquickhelper.pycode import ExtTestCase
from pymlbenchmark.benchmark.sklearn_helper import get_nb_skl_base_estimators
from pymlbenchmark.datasets import random_binary_classification


class TestSklearnHelper(ExtTestCase):

    def test_get_nb_skl_base_estimators(self):
        X, y = random_binary_classification(40, 4)
        rf = RandomForestClassifier(max_depth=2, n_estimators=4)
        rf.fit(X, y)
        n1 = get_nb_skl_base_estimators(rf, fitted=False)
        n2 = get_nb_skl_base_estimators(rf, fitted=True)
        self.assertEqual(n1, 2)
        self.assertEqual(n2, 5)


if __name__ == "__main__":
    unittest.main()
