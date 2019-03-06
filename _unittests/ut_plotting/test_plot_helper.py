# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import sys
import os
import unittest
import pandas
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


from src.pymlbenchmark.plotting.plot_helper import list_col_options, filter_df_options
from src.pymlbenchmark.plotting.plot_helper import options2label, ax_position, plt_colors


class TestPlotHelper(ExtTestCase):

    def test_list_col_options(self):
        df = pandas.DataFrame([
            dict(i=1, t='aa', x=0.5),
            dict(i=2, t='bb', x=0.5),
            dict(i=2, t='aa', x=0.5),
        ])

        res = list_col_options(df, ['i', 't'])
        self.assertEqual(res, [{'i': 1, 't': 'aa'},
                               {'i': 1, 't': 'bb'},
                               {'i': 2, 't': 'aa'},
                               {'i': 2, 't': 'bb'}])

        self.assertEqual(list_col_options(df, None), [None])

    def test_filter_df_options(self):
        df = pandas.DataFrame([
            dict(i=1, t='aa', x=0.5),
            dict(i=2, t='bb', x=0.5),
            dict(i=2, t='aa', x=0.5),
        ])

        sub = filter_df_options(df, list_col_options(df, ['i', 't'])[0])
        self.assertEqual(sub.shape, (1, 3))
        sub = filter_df_options(df, list_col_options(df, None))
        self.assertEqual(sub.shape, (3, 3))

    def test_options2label(self):
        res = options2label({'i': 1, 't': 'aa', 'x': 3.145667e10})
        self.assertEqual(res, "i=1 t=aa x=3.15e+10")

    def test_ax_position(self):
        self.assertEqual(ax_position((2, 2), (0, 0)), (0, 0))
        self.assertEqual(ax_position((1, 2), (0, 0)), (0, ))

    def test_plt_colors(self):
        r = plt_colors()
        self.assertIn('blue', r)
        self.assertIn('orange', r)


if __name__ == "__main__":
    unittest.main()
