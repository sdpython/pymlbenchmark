# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import sys
import os
import unittest
import pandas
from pyquickhelper.pycode import ExtTestCase, get_temp_folder


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


from src.pymlbenchmark.plotting import plot_bench_xtime


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


if __name__ == "__main__":
    unittest.main()
