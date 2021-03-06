# -*- coding: utf-8 -*-
"""
@brief      test log(time=6s)
"""
import os
import unittest
import pandas
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from pymlbenchmark.plotting import plot_bench_xtime


class TestPlotBenchScatter(ExtTestCase):

    def test_plot_logreg_xtime(self):
        from matplotlib import pyplot as plt
        temp = get_temp_folder(__file__, "temp_plot_logreg_xtime")
        img = os.path.join(temp, "plot_logreg.png")
        data = os.path.join(temp, "..", "data",
                            "onnxruntime_LogisticRegression.perf.csv")
        df = pandas.read_csv(data)
        ax = plot_bench_xtime(df, row_cols='N', col_cols='method',
                              hue_cols='fit_intercept',
                              title="unittest")
        fig = ax[0, 0].get_figure()
        fig.savefig(img)
        if __name__ == "__main__":
            plt.show()
        plt.close('all')
        self.assertExists(img)

    def test_plot_logreg_xtime_none(self):
        from matplotlib import pyplot as plt
        temp = get_temp_folder(__file__, "temp_plot_logreg_xtime")
        img = os.path.join(temp, "plot_logreg.png")
        data = os.path.join(temp, "..", "data",
                            "onnxruntime_LogisticRegression.perf.csv")
        df = pandas.read_csv(data)
        ax = plot_bench_xtime(df, row_cols='N', col_cols='method',
                              hue_cols=None,
                              title="unittest")
        fig = ax[0, 0].get_figure()
        fig.savefig(img)
        if __name__ == "__main__":
            plt.show()
        plt.close('all')
        self.assertExists(img)

    def test_plot_logreg_xtime_bug(self):
        from matplotlib import pyplot as plt
        temp = get_temp_folder(__file__, "temp_plot_logreg_xtime_bug")
        img = os.path.join(temp, "plot_cache.png")
        data = os.path.join(temp, "..", "data",
                            "bench_plot_gridsearch_cache.csv")
        df = pandas.read_csv(data)
        ax = plot_bench_xtime(df, row_cols=['n_jobs'],
                              x_value='mean',
                              hue_cols=['N'],
                              cmp_col_values='test',
                              title="unittest")
        fig = ax[0].get_figure()
        fig.savefig(img)
        if __name__ == "__main__":
            plt.show()
        plt.close('all')
        self.assertExists(img)


if __name__ == "__main__":
    unittest.main()
