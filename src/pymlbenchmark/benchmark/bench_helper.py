"""
@file
@brief Implements a benchmark about performance.
"""


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
