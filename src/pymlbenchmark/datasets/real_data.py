"""
@file
@brief Artificial datasets.
"""
import os
import pandas


def experiment_results(name):
    """
    Returns some real results mostly used in the documentation
    to produce graph to illustrate a function.

    @param      name    name of the dataset
    @return             dataframe

    List of available datasets:

    .. runpython::

        import os
        data = os.path.join(__WD__, 'data')
        for name in os.listdir(data):
            print(os.path.split(name)[0])

    Example of use:

    .. runpython::
        :showcode:

        from pymlbenchmark.datasets import experiment_results
        print(experiment_results('onnxruntime_LogisticRegression').head())
    """
    this = os.path.dirname(__file__)
    data = os.path.join(this, 'data', name + '.csv')
    if not os.path.exists(data):
        raise FileNotFoundError("Unable to find dataset '{}'.".format(name))
    return pandas.read_csv(data)
