# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
import pandas
from pyquickhelper.pycode import ExtTestCase
from pymlbenchmark.plotting.plot_helper import list_col_options, filter_df_options
from pymlbenchmark.plotting.plot_helper import options2label, ax_position, plt_colors, plt_styles


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

    def test_plt_styles(self):
        r = plt_styles()
        self.assertEqual(r[0], ('o', '-'))


if __name__ == "__main__":
    unittest.main()
