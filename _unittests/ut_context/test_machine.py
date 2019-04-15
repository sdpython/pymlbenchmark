# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from pymlbenchmark.context import machine_information


class TestContextMachine(ExtTestCase):

    def test_machine_information(self):
        res = machine_information()
        self.assertIsInstance(res, list)
        self.assertGreater(len(res), 3)
        for r in res:
            self.assertIn('name', r)
        self.assertEqual(res[0]['name'], 'date')

    def test_machine_information_pack(self):
        res = machine_information({'numpy'})
        self.assertIsInstance(res, list)
        self.assertGreater(len(res), 3)
        nb = 0
        for r in res:
            self.assertIn('name', r)
            if r['name'] == 'numpy':
                nb += 1
        self.assertEqual(nb, 1)


if __name__ == "__main__":
    unittest.main()
