"""
@file
@brief Plotting for benchmarks.
"""
from .plot_helper import list_col_options, filter_df_options, options2label
from .plot_helper import ax_position, plt_colors, plt_styles, move_color, move_color_add
from ..benchmark.bench_helper import remove_almost_nan_columns


def plot_bench_results(df, row_cols=None, col_cols=None, hue_cols=None,  # pylint: disable=R0914
                       cmp_col_values=('lib', 'skl'),
                       x_value='N', y_value='mean',
                       err_value=('lower', 'upper'),
                       title=None, box_side=4, labelsize=8,
                       fontsize="small", label_fct=None,
                       color_fct=None, ax=None):
    """
    Plots benchmark results.

    @param      df              benchmark results
    @param      row_cols        dataframe columns for graph rows
    @param      col_cols        dataframe columns for graph columns
    @param      hue_cols        dataframe columns for other options
    @param      cmp_col_values  if can be one column or one tuple ``(column, baseline name)``
    @param      x_value         value for x-axis
    @param      y_value         value to plot on y-axis (such as *mean*, *min*, ...)
    @param      err_value       lower and upper bounds
    @param      title           graph title
    @param      box_side        graph side, the function adjusts the size of the graph
    @param      labelsize       size of the labels
    @param      fontsize        font size see `Text properties
                                <https://matplotlib.org/api/text_api.html#matplotlib.text.Text>`_
    @param      label_fct       if not None, it is a function which
                                modifies the label before printing it on the graph
    @param      color_fct       if not None, it is a function which modifies
                                a color based on the label and the previous color
    @param      ax              existing axis
    @return                     fig, ax

    .. exref::
        :title: Plot benchmark results

        .. plot::

            from pymlbenchmark.datasets import experiment_results
            from pymlbenchmark.plotting import plot_bench_results
            import matplotlib.pyplot as plt

            df = experiment_results('onnxruntime_LogisticRegression')

            plot_bench_results(df, row_cols='N', col_cols='method',
                               x_value='dim', hue_cols='fit_intercept',
                               title="LogisticRegression\\nBenchmark scikit-learn / onnxruntime")
            plt.show()
    """
    if label_fct is None:
        def label_fct_(x):
            return x
        label_fct = label_fct_

    if color_fct is None:
        def color_fct_(la, col):
            return col
        color_fct = color_fct_

    import matplotlib.pyplot as plt  # pylint: disable=C0415
    if not isinstance(row_cols, (tuple, list)):
        row_cols = [row_cols]
    if not isinstance(col_cols, (tuple, list)):
        col_cols = [col_cols]
    if not isinstance(hue_cols, (tuple, list)):
        hue_cols = [hue_cols]

    all_cols = set(_ for _ in (
        row_cols + col_cols + hue_cols) if _ is not None)
    df = remove_almost_nan_columns(df, keep=all_cols)
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
    styles = plt_styles()

    nb_empty = 0
    nb_total = 0
    for row, row_opt in enumerate(lrows_options):

        sub = filter_df_options(df, row_opt)
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

            for color, hue_opt in zip(colors, lhues_options):
                ds = filter_df_options(sub2, hue_opt)
                if ds.shape[0] == 0:
                    continue
                legh = options2label(hue_opt, sep="\n")

                if isinstance(cmp_col_values, tuple):
                    y_cols = [x_value, cmp_col_values[0], y_value]
                    if err_value is not None:
                        lower_cols = [x_value, cmp_col_values[0], err_value[0]]
                        upper_cols = [x_value, cmp_col_values[0], err_value[1]]
                else:
                    y_cols = [x_value, cmp_col_values, y_value]
                    if err_value is not None:
                        lower_cols = [x_value, cmp_col_values, err_value[0]]
                        upper_cols = [x_value, cmp_col_values, err_value[1]]

                try:
                    piv = ds.pivot(*y_cols)
                except ValueError as e:
                    raise ValueError("Unable to compute a pivot on columns {}\nAvailable: {}\n{}".format(
                        y_cols, list(df.columns), ds[y_cols].head(n=10))) from e
                except KeyError as e:
                    raise ValueError(
                        "Unable to find columns {} in {}".format(y_cols, ds.columns))
                if lower_cols is not None:
                    try:
                        lower_piv = ds.pivot(*lower_cols)
                    except ValueError as e:
                        raise ValueError("Unable to compute a pivot on columns {}\n{}".format(
                            lower_cols, ds[lower_cols].head())) from e
                    except KeyError as e:
                        raise ValueError("Unable to find columns {} in {}".format(
                            lower_cols, ds.columns))
                else:
                    lower_piv = None
                if upper_cols is not None:
                    try:
                        upper_piv = ds.pivot(*upper_cols)
                    except ValueError as e:
                        raise ValueError("Unable to compute a pivot on columns {}\n{}".format(
                            upper_cols, ds[upper_cols].head())) from e
                    except KeyError as e:
                        raise ValueError("Unable to find columns {} in {}".format(
                            upper_cols, ds.columns))
                else:
                    upper_piv = None
                ys = list(piv.columns)

                piv = piv.reset_index(drop=False)
                if upper_piv is not None:
                    upper_piv = upper_piv.reset_index(drop=False)
                if lower_piv is not None:
                    lower_piv = lower_piv.reset_index(drop=False)

                for i, ly in enumerate(ys):
                    if hue_opt is None:
                        color = colors[i]
                    la = "{}-{}".format(ly, legh) if legh != '-' else ly
                    color_ = color_fct(la, color)
                    if upper_piv is not None and lower_piv is not None:
                        a.fill_between(piv[x_value], lower_piv[ly], upper_piv[ly],
                                       color=color_, alpha=0.1)
                    elif upper_piv is not None:
                        a.fill_between(piv[x_value], piv[ly], upper_piv[ly],
                                       color=color_, alpha=0.1)
                    elif lower_piv is not None:
                        a.fill_between(piv[x_value], lower_piv[ly], piv[ly],
                                       color=color_, alpha=0.1)

                for i, (ly, style) in enumerate(zip(ys, styles)):
                    if hue_opt is None:
                        color = colors[i]
                    la = "{}-{}".format(ly, legh) if legh != '-' else ly
                    color_ = color_fct(la, color)
                    lw = 4. if ly == cmp_col_values[1] else 1.5
                    ms = lw * 3
                    nc_add = move_color_add(style[0])
                    nc = move_color(color_, nc_add)
                    piv.plot(x=x_value, y=ly, ax=a, marker=style[0],
                             style=style[1], logx=True, logy=True,
                             c=nc, lw=lw, ms=ms,
                             label=label_fct(la))

            a.legend(loc=0, fontsize=fontsize)
            a.set_xlabel("{}\n{}".format(x_value, legx)
                         if row == shape[0] - 1 else "",
                         fontsize=fontsize)
            a.set_ylabel("{}\n{}".format(legy, y_value)
                         if col == 0 else "", fontsize=fontsize)
            if row == 0:
                a.set_title(legx, fontsize=fontsize)
            a.tick_params(labelsize=labelsize)
            for tick in a.yaxis.get_majorticklabels():
                tick.set_fontsize(labelsize)
            for tick in a.xaxis.get_majorticklabels():
                tick.set_fontsize(labelsize)
            plt.setp(a.get_xminorticklabels(), visible=False)
            plt.setp(a.get_yminorticklabels(), visible=False)

    if nb_empty == nb_total:
        raise RuntimeError("All graphs are empty for dataframe,\nrow_cols={},"
                           "\ncol_cols={},\nhue_cols={},\ncolumns={}".format(
                               row_cols, col_cols, hue_cols, df.columns))
    if title is not None:
        fig.suptitle(title, fontsize=labelsize)
    return ax
