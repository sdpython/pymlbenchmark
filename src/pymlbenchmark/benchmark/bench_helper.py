"""
@file
@brief Implements a benchmark about performance.
"""
import pandas
from pandas.api.types import is_numeric_dtype


def enumerate_options(options, filter_fct=None):
    """
    Enumerates all possible options.

    @param      options     dictionary ``{name: list of values}``
    @param      filter_fct  filters out some configurations
    @return                 list of dictionary ``{name: value}``

    .. runpython::
        :showcode:

        from pymlbenchmark.benchmark.bench_helper import enumerate_options
        options = dict(c1=[0, 1], c2=["aa", "bb"])
        for row in enumerate_options(options):
            print("no-filter", row)

        def filter_out(**opt):
            return not (opt["c1"] == 1 and opt["c2"] == "aa")

        for row in enumerate_options(options, filter_out):
            print("filter", row)
    """
    keys = list(sorted(options))
    mx = [len(options[k]) for k in keys]
    if min(mx) == 0:
        mi = min(zip(mx, keys))
        raise ValueError("Parameter '{0}' has no values.".format(mi[1]))
    pos = [0 for _ in keys]
    while pos[0] < mx[0]:
        opts = {k: options[k][pos[i]] for i, k in enumerate(keys)}
        if filter_fct is None or filter_fct(**opts):
            yield opts
        p = len(pos) - 1
        pos[p] += 1
        while p > 0 and pos[p] >= mx[p]:
            pos[p] = 0
            p -= 1
            pos[p] += 1


def bench_pivot(data, experiment='lib', value='mean', index=None):
    """
    Merges all results for one set of parameters in one row.

    @param  data        :epkg:`DataFrame`
    @param  experiment  column which identifies an experiment
    @param  value       value to plot
    @param  index       set of parameters which identifies
                        an experiment, if None, guesses it
    @return             :epkg:`DataFrame`

    .. runpython::
        :showcode:

        import pandas
        from pymlbenchmark.datasets import experiment_results
        from pymlbenchmark.benchmark.bench_helper import bench_pivot

        df = experiment_results('onnxruntime_LogisticRegression')
        piv = bench_pivot(df)
        print(piv.head())
    """
    if not isinstance(experiment, list):
        experiment = [experiment]
    if index is None:
        metrics = ['lower', 'max', 'max3', 'mean',
                   'median', 'min', 'min3', 'repeat', 'upper']
        data = data.copy()
        nonan = []
        for c in data.columns:
            if c in metrics or c in experiment:
                continue
            if c.endswith('_nodes') or c.endswith('_size'):
                continue
            nn = sum(data[c].isnull())
            if nn == data.shape[0]:
                continue
            if nn == 0:
                nonan.append(c)
                continue
            if is_numeric_dtype(data[c]):
                data[c].fillna(-1, inplace=True)
            else:
                data[c].fillna("", inplace=True)
            print(c, len(set(c)), is_numeric_dtype(data[c]))
            nonan.append(c)
        index = nonan
    keep = list(index)
    if isinstance(value, str):
        keep.append(value)
    else:
        keep.extend(value)
    keep.extend(experiment)
    for c in keep:
        if c not in data.columns:
            raise ValueError(
                "Unable to find '{}' in {}.".format(c, data.columns))
    data_short = data[keep]
    gr = data_short.groupby(index + experiment).count()
    if gr[value].max() >= 2:
        gr = gr[gr[value] > 1]
        raise ValueError(
            "The set of parameters does not identify an experiment."
            "\nindex: {}\nexperiment: {}\nvalue: {}\ncolumns: {}\n--\n{}".format(
                index, experiment, value, data.columns, gr[gr[value] >= 2]))
    piv = pandas.pivot_table(data_short, values=value, index=index, columns=experiment,
                             aggfunc='mean', dropna=False)
    return piv


def remove_almost_nan_columns(df, keep=None, fill_keep=True):
    """
    Automatically removes columns with more than 1/3
    nan values.

    @param      df          dataframe
    @param      keep        columns to skip
    @param      fill_keep   if not None, fill nan value
    @return                 clean dataframe
    """
    if keep is None:
        keep = set()
    n = df.shape[0] * 1 // 3
    nanc = [c for c in df.columns if c not in keep and sum(
        df[c].isnull()) >= n]
    if keep and fill_keep:
        df = df.copy()
        for c in keep:
            if is_numeric_dtype(df[c]):
                df[c].fillna(-1, inplace=True)
            else:
                df[c] = df[c].astype(str)
                df[c].fillna("", inplace=True)
    if nanc:
        return df.drop(nanc, axis=1)
    return df
