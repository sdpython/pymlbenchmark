# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import os
import unittest
import pandas
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from pymlbenchmark.plotting import plot_bench_results


class TestPlotBenchCurve(ExtTestCase):

    def test_plot_logreg(self):
        from matplotlib import pyplot as plt
        temp = get_temp_folder(__file__, "temp_plot_logreg")
        img = os.path.join(temp, "plot_logreg.png")
        data = os.path.join(temp, "..", "data",
                            "onnxruntime_LogisticRegression.perf.csv")
        df = pandas.read_csv(data)
        ax = plot_bench_results(df, row_cols='N', col_cols='method',
                                x_value='dim', hue_cols='fit_intercept',
                                title="unittest")
        fig = ax[0, 0].get_figure()
        fig.savefig(img)
        if __name__ == "__main__":
            plt.show()
        plt.close('all')
        self.assertExists(img)

    def test_plot_polypf(self):
        from matplotlib import pyplot as plt
        temp = get_temp_folder(__file__, "temp_plot_polypf")
        img = os.path.join(temp, "plot_polypf.png")
        data = os.path.join(temp, "..", "data",
                            "plot_bench_polynomial_features_partial_fit.perf.csv")
        df = pandas.read_csv(data)
        ax = plot_bench_results(df, row_cols='N', col_cols=None,
                                x_value='dim', hue_cols=None,
                                cmp_col_values='test',
                                title="unittest")
        fig = ax[0].get_figure()
        fig.savefig(img)
        if __name__ == "__main__":
            plt.show()
        plt.close('all')
        self.assertExists(img)

    def test_plot_polypf2(self):
        from matplotlib import pyplot as plt
        temp = get_temp_folder(__file__, "temp_plot_polypf2")
        img = os.path.join(temp, "plot_polypf2.png")
        data = os.path.join(temp, "..", "data",
                            "plot_bench_polynomial_features.perf.csv")
        df = pandas.read_csv(data)
        ax = plot_bench_results(df, row_cols=['N', 'order'],
                                col_cols=['degree'], x_value='dim',
                                hue_cols=['interaction_only'],
                                cmp_col_values='test',
                                title="unittest")
        fig = ax[0, 0].get_figure()
        fig.savefig(img)
        if __name__ == "__main__":
            plt.show()
        plt.close('all')
        self.assertExists(img)


if __name__ == "__main__":
    unittest.main()
