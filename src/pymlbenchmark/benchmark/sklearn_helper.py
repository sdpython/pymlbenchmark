"""
@file
@brief Helpers about :epkg:`scikit-learn`.
"""
from sklearn.base import BaseEstimator


def get_nb_skl_base_estimators(obj, fitted=True):
    """
    Returns the number of :epkg:`scikit-learn` *BaseEstimator*
    including in a pipeline. The function assumes the pipeline
    is not recursive.

    @param      obj         object to walk through
    @param      fitted      count the number of fitted object
    @return                 number of base estimators including this one
    """
    ct = 0
    if isinstance(obj, BaseEstimator):
        ct += 1
        for k, o in obj.__dict__.items():
            if k in {'base_estimator_'}:
                continue
            t = 0
            if fitted:
                if k.endswith('_'):
                    t = get_nb_skl_base_estimators(o, fitted=fitted)
            elif not k.endswith('_'):
                t = get_nb_skl_base_estimators(o, fitted=fitted)
            ct += t
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            ct += get_nb_skl_base_estimators(o, fitted=fitted)
    elif isinstance(obj, dict):
        for o in obj.values():
            ct += get_nb_skl_base_estimators(o, fitted=fitted)
    return ct
