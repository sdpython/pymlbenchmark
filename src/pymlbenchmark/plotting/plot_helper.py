"""
@file
@brief Plotting helpers
"""
import numpy
from ..benchmark import enumerate_options


def list_col_options(df, cols):
    """
    @param      df      dataframe
    @param      cols    cols to look for options
    @return             options

    .. exref::
        :title: Enumerate all coordinates

        .. runpython::
            :showcode:

            from pymlbenchmark.plotting.plot_helper import list_col_options
            from pandas import DataFrame
            df = DataFrame([
                dict(i=1, t='aa', x=0.5),
                dict(i=2, t='bb', x=0.5),
                dict(i=2, t='aa', x=0.5),
            ])
            for opt in list_col_options(df, ['i', 't']):
                print(opt)

            # if None...
            print(list_col_options(df, None))
    """
    if cols is None:
        return [None]

    if not isinstance(cols, (tuple, list)):
        cols = [cols]
    elif cols[0] is None:
        return [None]

    options = {k: list(sorted(set(df[k]))) for k in cols}
    return list(enumerate_options(options))


def filter_df_options(df, options):
    """
    Filters out rows from a dataframe.

    @param      df          dataframe
    @param      options     options
    @return                 filtered dataframe

    .. runpython::
        :showcode:

        from pymlbenchmark.plotting.plot_helper import filter_df_options
        from pandas import DataFrame
        df = DataFrame([
            dict(i=1, t='aa', x=0.5),
            dict(i=2, t='bb', x=0.5),
            dict(i=2, t='aa', x=0.5),
        ])

        sub = filter_df_options(df, {'i': 1, 't': 'aa'})
        print(sub)

        sub = filter_df_options(df, [None])
        print(sub)
    """
    if options is None:
        return df
    elif isinstance(options, list):
        if options[0] is None:
            return df
        else:
            raise RuntimeError(
                "options must be dictionary or [None] not {}".format(options))
    for k, v in options.items():
        df = df[df[k] == v]
    return df


def options2label(opt, sep=" ", format_number="{:.3g}"):
    """
    Converts a list of options into a label.

    @param      opt             dictionary
    @param      sep             separator
    @param      format_number   how to format real numbers
    @return                     string

    .. runpython::
        :showcode:

        from pymlbenchmark.plotting.plot_helper import options2label
        res = options2label({'i': 1, 't': 'aa', 'x': 3.145667e10})
        print(res)
    """
    if opt is None:
        return "-"
    rows = []
    for k, v in sorted(opt.items()):
        if isinstance(v, (float, numpy.float64, numpy.float32)):
            v = format_number.format(v)
        rows.append("{}={}".format(k, v))
    return sep.join(rows)


def ax_position(shape, pos):
    """
    :epkg:`matplotlib` uses a one dimension
    array if the number of columns or rows is 1.
    This function makes a correction.

    @param      shape       graph shape
    @param      pos         graph current position
    @return                 corrected position

    .. runpython::
        :showcode:

        from pymlbenchmark.plotting.plot_helper import ax_position
        print(ax_position((2, 2), (0, 0)))
        print(ax_position((1, 2), (0, 0)))
    """
    res = []
    for a, b in zip(shape, pos):
        if a > 1:
            res.append(b)
    return tuple(res)


def plt_colors():
    """
    Returns :epkg:`matplotlib` colors.

    .. runpython::
        :showcode:

        from pymlbenchmark.plotting.plot_helper import plt_colors
        print(plt_colors())
    """
    import matplotlib.colors as mcolors
    return [k.split(':')[-1] for k in mcolors.TABLEAU_COLORS]


def plt_styles():
    """
    Returns :epkg:`matplotlib` styles.

    .. runpython::
        :showcode:

        from pymlbenchmark.plotting.plot_helper import plt_styles
        print(plt_styles())
    """
    return [('o', '-'), ('x', '-'), ('*', '-'), ('^', '-'),
            ('o', '--'), ('x', '--'), ('*', '--'), ('^', '--')]
