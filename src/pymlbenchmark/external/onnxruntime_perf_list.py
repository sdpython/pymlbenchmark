"""
@file
@brief Returns predefined tests.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from .onnxruntime_perf import OnnxRuntimeBenchPerfTestBinaryClassification


def onnxruntime_perf_binary_classifiers():
    """
    Returns a list of benchmarks for binary classifier.
    It compares :epkg:`onnxruntime` predictions
    against :epkg:`scikit-learn`.
    """
    dims = [1, 5, 10, 20, 50, 100, 150]
    N = [1, 10]
    max_depths = [2, 5, 10, 15, 20]

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
        # bayes
        {'fct': lambda **opts: bincl(MultinomialNB, **opts),
         'pbefore': dict(dim=dims, alpha=[0., 0.5, 1.], fit_prior=[True, False]),
         'pafter': dict(N=N),
         'name': 'MultinomialNB'},
        {'fct': lambda **opts: bincl(BernoulliNB, **opts),
         'pbefore': dict(dim=dims, alpha=[0., 0.5, 1.],
                         binarize=[0., 0.5, 1.],
                         fit_prior=[True, False]),
         'pafter': dict(N=N),
         'name': 'BernoulliNB'},
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
