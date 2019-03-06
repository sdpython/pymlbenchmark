"""
@file
@brief Plotting for benchmarks.
"""
from .plot_helper import list_col_options, filter_df_options, options2label, ax_position, plt_colors


def plot_bench_results(df, row_cols=None, col_cols=None, hue_cols=None,
                       cmp_col_values=('lib', ('skl', 'ort')),
                       x_value='N', y_value='mean', title=None,
                       fig=None, ax=None, box_side=4):
    """
    Plots benchmark results.

    @param      df              benchmark results
    @param      row_cols        dataframe columns for graph rows
    @param      col_cols        dataframe columns for graph columns
    @param      hue_cols
    @param      cmp_col_values
    @param      col_value
    @param      ax              existing axis
    @param      fig             existing figure
    @param      box_side        graph side, the function adjusts the size of the graph
    @return                     fig, ax
    """
    if not isinstance(row_cols, (tuple, list)):
        row_cols = [row_cols]
    if not isinstance(col_cols, (tuple, list)):
        col_cols = [col_cols]
    if not isinstance(hue_cols, (tuple, list)):
        hue_cols = [hue_cols]

    lrows_options = list_col_options(df, row_cols)
    lcols_options = list_col_options(df, col_cols)
    lhues_options = list_col_options(df, hue_cols)

    shape = (len(lrows_options), len(lcols_options))
    if ax is None:
        if fig is not None:
            fig, ax = plt.subplots(shape[0], shape[1])
        else:
            import matplotlib.pyplot as plt
            figsize = (shape[0] * box_side, shape[1] * box_side)
            fig, ax = plt.subplots(shape[0], shape[1], figsize=figsize)
    elif ax.shape != shape:
        raise RuntimeError(
            "Shape mismatch ax.shape={} when expected values is {}".format(ax.shape, shape))
    elif fig is not None:
        raise NotImplementedError("ax is not None, fig must not be None")
    colors = plt_colors()

    for row, row_opt in enumerate(lrows_options):

        sub = filter_df_options(df, row_opt)
        if sub.shape[0] == 0:
            continue
        legy = options2label(row_opt)

        for col, col_opt in enumerate(lcols_options):
            sub2 = filter_df_options(sub, col_opt)
            if sub2.shape[0] == 0:
                continue
            legx = options2label(col_opt)

            pos = ax_position(shape, (row, col))
            a = ax[pos] if pos else ax

            for color, hue_opt in zip(colors, lhues_options):
                ds = filter_df_options(sub2, hue_opt)
                if ds.shape[0] == 0:
                    continue
                legh = options2label(hue_opt)

                keep_cols = [x_value, cmp_col_values[0], y_value]
                try:
                    piv = ds.pivot(*keep_cols)
                except ValueError as e:
                    raise ValueError("Unable to compute a pivot on columns {}\n{}".format(
                        keep_cols, ds[keep_cols].head())) from e
                ys = list(piv.columns)
                piv = piv.reset_index(drop=False)

                for ly in ys:
                    style = '--' if ly == cmp_col_values[1][0] else '-'
                    piv.plot(x=x_value, y=ly, ax=a, style=style,
                             logx=True, logy=True, c=color,
                             label="{}-{}".format(ly, legh))

            a.legend(loc=0, fontsize='x-small')
            a.set_xlabel("{}\n{}".format(x_value, legx) if row ==
                         shape[0] - 1 else "", fontsize='x-small')
            a.set_ylabel("{}\n{}".format(legy, y_value)
                         if col == 0 else "", fontsize='x-small')
            if row == 0:
                a.set_title(legx, fontsize='x-small')
            a.tick_params(labelsize=7)
            for tick in a.yaxis.get_majorticklabels():
                tick.set_fontsize(7)
            for tick in a.xaxis.get_majorticklabels():
                tick.set_fontsize(7)

    if title is not None:
        fig.suptitle(title, fontsize=10)
    return fig, ax
