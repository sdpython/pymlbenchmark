"""
@file
@brief Plotting for benchmarks.
"""
from .plot_helper import list_col_options, filter_df_options, options2label
from .plot_helper import ax_position, plt_colors
from ..benchmark.bench_helper import remove_almost_nan_columns


def plot_bench_xtime(df, row_cols=None, col_cols=None, hue_cols=None,
                     cmp_col_values=('lib', 'skl'),
                     x_value='mean', y_value='xtime',
                     parallel=(1., 0.5), title=None,
                     box_side=4, labelsize=8, ax=None):
    """
    Plots benchmark acceleration.

    @param      df              benchmark results
    @param      row_cols        dataframe columns for graph rows
    @param      col_cols        dataframe columns for graph columns
    @param      hue_cols        dataframe columns for other options
    @param      cmp_col_values  if can be one column or one tuple
                                ``(column, baseline name)``
    @param      x_value         value for x-axis
    @param      y_value         value to plot on y-axis (such as *mean*, *min*, ...)
    @param      parallel        lower and upper bounds
    @param      title           graph title
    @param      box_side        graph side, the function adjusts the
                                size of the graph
    @param      labelsize       size of the labels
    @param      ax              existing axis
    @return                     fig, ax

    .. exref::
        :title: Plot benchmark improvments

        .. plot::

            from pymlbenchmark.datasets import experiment_results
            from pymlbenchmark.plotting import plot_bench_xtime
            import matplotlib.pyplot as plt

            df = experiment_results('onnxruntime_LogisticRegression')

            plot_bench_xtime(df, row_cols='N', col_cols='method',
                             hue_cols='fit_intercept',
                             title="LogisticRegression\\nAcceleration scikit-learn / onnxruntime")
            plt.show()
    """
    import matplotlib.pyplot as plt
    if not isinstance(row_cols, (tuple, list)):
        row_cols = [row_cols]
    if not isinstance(col_cols, (tuple, list)):
        col_cols = [col_cols]
    if not isinstance(hue_cols, (tuple, list)):
        hue_cols = [hue_cols]

    df = remove_almost_nan_columns(df)
    lrows_options = list_col_options(df, row_cols)
    lcols_options = list_col_options(df, col_cols)
    lhues_options = list_col_options(df, hue_cols)

    shape = (len(lrows_options), len(lcols_options))
    if ax is None:
        figsize = (shape[1] * box_side, shape[0] * box_side)
        fig, ax = plt.subplots(shape[0], shape[1], figsize=figsize)
    elif ax.shape != shape:
        raise RuntimeError(
            "Shape mismatch ax.shape={} when expected values is {}".format(ax.shape, shape))
    else:
        fig = plt.gcf()
    colors = plt_colors()

    dropc = "lower,max,max3,mean,median,min,min3,repeat,upper".split(',')
    dropc = [c for c in dropc if c not in [x_value, y_value]]
    df = df.drop(dropc, axis=1)
    index = [c for c in df.columns if c not in [
        x_value, y_value, cmp_col_values[0]]]
    piv = df.pivot_table(index=index, values=x_value,
                         columns=cmp_col_values[0])
    piv = piv.reset_index(drop=False)
    if piv.shape[0] == 0:
        raise RuntimeError("pivot table is empty,\nindex={},\nx_value={},\ncolumns={},\ndf.columns={}".format(
            index, x_value, cmp_col_values[0], df.columns))
    vals = list(sorted(set(df[cmp_col_values[0]])))

    nb_empty = 0
    nb_total = 0
    for row, row_opt in enumerate(lrows_options):

        sub = filter_df_options(piv, row_opt)
        nb_total += 1
        if sub.shape[0] == 0:
            nb_empty += 1
            continue
        legy = options2label(row_opt, sep="\n")

        for col, col_opt in enumerate(lcols_options):
            sub2 = filter_df_options(sub, col_opt)
            if sub2.shape[0] == 0:
                continue
            legx = options2label(col_opt, sep="\n")

            pos = ax_position(shape, (row, col))
            a = ax[pos] if pos else ax

            if parallel is not None:
                mi, ma = sub2[cmp_col_values[1]].min(
                ), sub2[cmp_col_values[1]].max()
                for p in parallel:
                    style = '-' if p == 1 else "--"
                    a.plot([mi, ma], [p, p], style, color='black',
                           label="%1.1fx" % (1. / p))

            ic = 0
            for color, hue_opt in zip(colors, lhues_options):
                ds = filter_df_options(sub2, hue_opt).copy()
                if ds.shape[0] == 0:
                    continue
                legh = options2label(hue_opt, sep="\n")

                im = 0
                for ly in vals:
                    if ly == cmp_col_values[1]:
                        continue

                    ds["xtime"] = ds[ly] / ds[cmp_col_values[1]]
                    if hue_opt is None:
                        color = colors[ic % len(colors)]
                        ic += 1
                    if ly == cmp_col_values[1]:
                        marker = 'o'
                    else:
                        marker = '.x+'[im]
                        im += 1
                    ds.plot(x=cmp_col_values[1], y=y_value, ax=a, marker=marker,
                            logx=True, logy=True, c=color, lw=2,
                            label="{}-{}".format(ly, legh)
                                  if legh != '-' else ly,
                            kind="scatter")

            a.set_xlabel("{}\n{}".format(x_value, legx)
                         if row == shape[0] - 1 else "",
                         fontsize='x-small')
            a.set_ylabel("{}\n{}".format(legy, y_value)
                         if col == 0 else "", fontsize='x-small')

            a.legend(loc=0, fontsize='x-small')
            a.tick_params(labelsize=labelsize)
            for tick in a.yaxis.get_majorticklabels():
                tick.set_fontsize(labelsize)
            for tick in a.xaxis.get_majorticklabels():
                tick.set_fontsize(labelsize)
            plt.setp(a.get_xminorticklabels(), visible=False)
            plt.setp(a.get_yminorticklabels(), visible=False)

    if nb_empty == nb_total:
        raise RuntimeError("All graphs are empty for dataframe,\nrow_cols={},\ncol_cols={},\nhue_cols={},\ncolumns={}".format(
            row_cols, col_cols, hue_cols, df.columns))
    if title is not None:
        fig.suptitle(title, fontsize=10)
    return ax