"""
@file
@brief Implements a benchmark about performance for :epkg:`onnxruntime`
"""
import contextlib
from io import BytesIO, StringIO
import numpy
from numpy.testing import assert_almost_equal
from sklearn.ensemble.forest import BaseForest
from sklearn.tree.tree import BaseDecisionTree
from ..benchmark import BenchPerfTest
from ..benchmark.sklearn_helper import get_nb_skl_base_estimators
from ..datasets import random_binary_classification


class OnnxRuntimeBenchPerfTestBinaryClassification(BenchPerfTest):
    """
    Specific test to compare computing time predictions
    with :epkg:`scikit-learn` and :epkg:`onnxruntime`.
    See example :ref:`l-example-onnxruntime-logreg`.
    The class requires the following modules to be installed:
    :epkg:`onnx`, :epkg:`onnxruntime`, :epkg:`skl2onnx`.
    """

    def __init__(self, classifier, dim=None, **opts):
        """
        @param      classifier  classifier class
        @param      opts        training settings
        @param      dim         number of features
        """
        # These libraries are optional.
        from skl2onnx import convert_sklearn  # pylint: disable=E0401
        from skl2onnx.common.data_types import FloatTensorType  # pylint: disable=E0401
        from onnxruntime import InferenceSession  # pylint: disable=E0401

        if dim is None:
            raise RuntimeError("dim must be defined.")
        BenchPerfTest.__init__(self, **opts)

        allowed = {"max_depth"}
        opts = {k: v for k, v in opts.items() if k in allowed}
        self.skl = classifier(**opts)
        X, y = random_binary_classification(100000, dim)
        self.skl.fit(X, y)

        initial_types = [('X', FloatTensorType([1, X.shape[1]]))]
        self.logconvert = StringIO()
        with contextlib.redirect_stdout(self.logconvert):
            with contextlib.redirect_stderr(self.logconvert):
                onx = convert_sklearn(self.skl, initial_types=initial_types)
        f = BytesIO()
        f.write(onx.SerializeToString())
        self.ort_onnx = onx
        content = f.getvalue()
        self.ort = InferenceSession(content)
        self.outputs = [o.name for o in self.ort.get_outputs()]
        self.extract_model_info_skl()
        self.extract_model_info_ort(ort_size=len(content))

    def extract_model_info_skl(self, **kwargs):
        """
        Populates member ``self.skl_info`` with additional
        information on the model such as the number of node for
        a decision tree.
        """
        self.skl_info = dict(
            skl_nb_nase_estimators=get_nb_skl_base_estimators(self.skl, fitted=True))
        self.skl_info.update(kwargs)
        if isinstance(self.skl, BaseDecisionTree):
            self.skl_info["skl_dt_nodes"] = self.skl.tree_.node_count
        elif isinstance(self.skl, BaseForest):
            self.skl_info["skl_rf_nodes"] = sum(
                est.tree_.node_count for est in self.skl.estimators_)

    def extract_model_info_ort(self, **kwargs):
        """
        Populates member ``self.ort_info`` with additional
        information on the :epkg:`ONNX` graph.
        """
        self.ort_info = dict(ort_nodes=len(
            self.ort_onnx.graph.node))  # pylint: disable=E1101
        self.ort_info.update(kwargs)

    def data(self, N=None, dim=None, **kwargs):  # pylint: disable=W0221
        """
        Generates random features.

        @param      N       number of observations
        @param      dim     number of features
        """
        if dim is None:
            raise RuntimeError("dim must be defined.")
        if N is None:
            raise RuntimeError("N must be defined.")
        return random_binary_classification(N, dim)[:1]

    def fcts(self, dim=None, **kwargs):  # pylint: disable=W0221
        """
        Returns four functions, tests methods
        *perdict*, *predict_proba* for both
        :epkg:`scikit-learn` and :epkg:`onnxruntime`.
        """
        def predict_skl_predict(X, model=self.skl):
            return model.predict(X)

        def predict_skl_predict_proba(X, model=self.skl):
            return model.predict_proba(X)

        def predict_onnxrt_predict(X, sess=self.ort, output=self.outputs):
            return numpy.array(sess.run(self.outputs[:1],
                                        {'X': X.astype(numpy.float32)}))

        def predict_onnxrt_predict_proba(X, sess=self.ort, output=self.outputs):
            res = sess.run(self.outputs[1:],
                           {'X': X.astype(numpy.float32)})[0]
            # do not use DataFrame to convert the output into array,
            # it takes too much time
            out = numpy.empty((len(res), len(res[0])), dtype=numpy.float32)
            for i, row in enumerate(res):
                for k, v in row.items():
                    out[i, k] = v
            return out

        fcts = [{'method': 'predict', 'lib': 'skl', 'fct': predict_skl_predict},
                {'method': 'predict', 'lib': 'ort', 'fct': predict_onnxrt_predict}]

        if hasattr(self.skl, '_check_proba'):
            try:
                self.skl._check_proba()
                prob = True
            except AttributeError:
                prob = False
        elif hasattr(self.skl, 'predict_proba'):
            prob = True
        else:
            prob = False

        if prob:
            fcts.extend([
                {'method': 'predict_proba', 'lib': 'skl',
                    'fct': predict_skl_predict_proba},
                {'method': 'predict_proba', 'lib': 'ort',
                    'fct': predict_onnxrt_predict_proba}
            ])
        for fct in fcts:
            if fct['lib'] == 'skl':
                fct.update(self.skl_info)
            elif fct['lib'] == 'ort':
                fct.update(self.ort_info)
        return fcts

    def validate(self, results):
        """
        Checks that methods *predict* and *predict_proba* returns
        the same results for both :epkg:`scikit-learn` and
        :epkg:`onnxruntime`.
        """
        for method in {'predict', 'predict_proba'}:
            res = [row[1] for row in results if row[0]['method'] == method]
            if len(res) > 0 and res[0].shape[0] <= 10000:
                for i in range(1, len(res)):
                    p1, p2 = res[0], res[i]
                    if len(p1.shape) == 1 and len(p2.shape) == 2:
                        p2 = p2.ravel()
                    try:
                        assert_almost_equal(p1, p2, decimal=4)
                    except AssertionError as e:
                        rows = [row[0]
                                for row in results if row[0]['method'] == method]
                        raise AssertionError("Dim {} - discrepencies between\n{} and\n{}.".format(
                            p1.shape, rows[0], rows[i])) from e

    def model_info(self, model):
        """
        Returns additional informations about a model.

        @param      model       model to describe
        @return                 dictionary with additional descriptor
        """
