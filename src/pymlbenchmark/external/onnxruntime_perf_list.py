"""
@file
@brief Returns predefined tests.
"""
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from .onnxruntime_perf import OnnxRuntimeBenchPerfTestBinaryClassification
from ..context import machine_information
from ..benchmark import BenchPerf


def onnxruntime_perf_binary_classifiers(bincl=None):
    """
    Returns a list of benchmarks for binary classifier.
    It compares :epkg:`onnxruntime` predictions
    against :epkg:`scikit-learn`.

    @param      bincl       test to chenge, by default, it is
                            @see cl OnnxRuntimeBenchPerfTestBinaryClassification
    """
    dims = [1, 5, 10, 20, 50, 100, 150]
    N = [1, 10]
    max_depths = [2, 5, 10, 15, 20]

    if bincl is None:
        bincl = OnnxRuntimeBenchPerfTestBinaryClassification

    return [
        {'fct': lambda **opts: bincl(LogisticRegression, **opts),
         'pbefore': dict(dim=dims, fit_intercept=[True, False]),
         'pafter': dict(N=N),
         'name': 'LogisticRegression'},
        # linear
        {'fct': lambda **opts: bincl(SGDClassifier, **opts),
         'pbefore': dict(dim=dims, average=[False, True],
                         loss=['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']), 'pafter': dict(N=N),
         'name': 'SGDClassifier'},
        # trees
        {'fct': lambda **opts: bincl(DecisionTreeClassifier, **opts),
         'pbefore': dict(dim=dims, max_depth=max_depths),
         'pafter': dict(N=N),
         'name': 'DecisionTreeClassifier'},
        {'fct': lambda **opts: bincl(RandomForestClassifier, **opts),
         'pbefore': dict(dim=dims, max_depth=max_depths, n_estimators=[1, 10, 100]),
         'pafter': dict(N=N),
         'name': 'RandomForestClassifier'},
    ]


def run_onnxruntime_test(folder, name, repeat=100, verbose=True,
                         stop_if_error=True, validate=True,
                         N=None, dim=None, fLOG=None):
    """
    Runs a benchmark for :epkg:`onnxruntime`.

    @param      folder          where to dump the results
    @param      name            name of the test (one in the list returned by
                                @see fn onnxruntime_perf_binary_classifiers)
    @param      repeat          number of times to repeat predictions
    @param      verbose         print progress with :epkg:`tqdm`
    @param      stop_if_error   by default, it stops when method *validate*
                                fails, if False, the function stores the exception
    @param      validate        validate the outputs against the baseline
    @param      N               overwrites *N* parameter
    @param      dim             overwrites *dims* parameter
    @param      fLOG            logging function
    @return                     two dataframes, one for the results,
                                the other one for the context (see @see fn machine_information)
    """
    import pandas
    if fLOG:
        fLOG("Start '%s'" % name)

    res = onnxruntime_perf_binary_classifiers()
    sel = [r for r in res if r['name'] == name]
    if len(sel) != 1:
        raise ValueError("Unable to find one test for '%s'." % name)
    res = sel[0]
    res = res.copy()
    if N is not None:
        res["pafter"]['N'] = N
    if dim is not None:
        res["pbefore"]['dim'] = dim

    bp = BenchPerf(res['pbefore'], res['pafter'], res['fct'])
    results = list(bp.enumerate_run_benchs(repeat=repeat, verbose=verbose,
                                           stop_if_error=stop_if_error,
                                           validate=validate))
    results_df = pandas.DataFrame(results)
    if folder:
        out = os.path.join(folder, "onnxruntime_%s.perf.csv" % name)
        results_df.to_csv(out, index=False)

    subset = {'sklearn', 'numpy', 'pandas', 'onnxruntime',
              'skl2onnx', 'onnxconverters_common'}

    df2 = pandas.DataFrame(machine_information(subset))
    if folder:
        out = os.path.join(folder, "onnxruntime_%s.time.csv" % name)
        df2.to_csv(out, index=False)
    if fLOG:
        fLOG("Done '%s'" % name)
    return results_df, df2
