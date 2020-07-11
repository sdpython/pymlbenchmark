"""
@file
@brief Implements a benchmark about performance for :epkg:`onnxruntime`
"""
import contextlib
from collections import OrderedDict
from io import BytesIO, StringIO
import numpy
from numpy.testing import assert_almost_equal
import pandas
try:
    from sklearn.ensemble._forest import BaseForest
except ImportError:  # pragma: no cover
    from sklearn.ensemble.forest import BaseForest
try:
    from sklearn.tree._classes import BaseDecisionTree
except ImportError:  # pragma: no cover
    from sklearn.tree.tree import BaseDecisionTree
from mlprodict.onnxrt import OnnxInference
from mlprodict.tools.asv_options_helper import (
    get_opset_number_from_onnx, get_ir_version_from_onnx)
from ..benchmark import BenchPerfTest
from ..benchmark.sklearn_helper import get_nb_skl_base_estimators


class OnnxRuntimeBenchPerfTest(BenchPerfTest):
    """
    Specific test to compare computing time predictions
    with :epkg:`scikit-learn` and :epkg:`onnxruntime`.
    See example :ref:`l-example-onnxruntime-logreg`.
    The class requires the following modules to be installed:
    :epkg:`onnx`, :epkg:`onnxruntime`, :epkg:`skl2onnx`,
    :epkg:`mlprodict`.
    """

    def __init__(self, estimator, dim=None, N_fit=100000,
                 runtimes=('python_compiled', 'onnxruntime1'),
                 onnx_options=None, dtype=numpy.float32,
                 **opts):
        """
        @param      estimator       estimator class
        @param      dim             number of features
        @param      N_fit           number of observations to fit an estimator
        @param      runtimes        runtimes to test for class :epkg:`OnnxInference`
        @param      opts            training settings
        @param      onnx_options    ONNX conversion options
        @param      dtype           dtype (float32 or float64)
        """
        # These libraries are optional.
        from skl2onnx import to_onnx  # pylint: disable=E0401,C0415
        from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType  # pylint: disable=E0401,C0415

        if dim is None:
            raise RuntimeError(  # pragma: no cover
                "dim must be defined.")
        BenchPerfTest.__init__(self, **opts)

        allowed = {"max_depth"}
        opts = {k: v for k, v in opts.items() if k in allowed}
        self.dtype = dtype
        self.skl = estimator(**opts)
        X, y = self._get_random_dataset(N_fit, dim)
        try:
            self.skl.fit(X, y)
        except Exception as e:  # pragma: no cover
            raise RuntimeError("X.shape={}\nopts={}\nTraining failed for {}".format(
                X.shape, opts, self.skl)) from e

        if dtype == numpy.float64:
            initial_types = [('X', DoubleTensorType([None, X.shape[1]]))]
        elif dtype == numpy.float32:
            initial_types = [('X', FloatTensorType([None, X.shape[1]]))]
        else:
            raise ValueError(  # pragma: no cover
                "Unable to convert the model into ONNX, unsupported dtype {}.".format(dtype))
        self.logconvert = StringIO()
        with contextlib.redirect_stdout(self.logconvert):
            with contextlib.redirect_stderr(self.logconvert):
                onx = to_onnx(self.skl, initial_types=initial_types,
                              options=onnx_options,
                              target_opset=get_opset_number_from_onnx())
                onx.ir_version = get_ir_version_from_onnx()

        self._init(onx, runtimes)

    def _get_random_dataset(self, N, dim):
        """
        Returns a random datasets.
        """
        raise NotImplementedError(  # pragma: no cover
            "This method must be overloaded.")

    def _init(self, onx, runtimes):
        "Finalizes the init."
        f = BytesIO()
        f.write(onx.SerializeToString())
        self.ort_onnx = onx
        content = f.getvalue()
        self.ort = OrderedDict()
        self.outputs = OrderedDict()
        for r in runtimes:
            self.ort[r] = OnnxInference(content, runtime=r)
            self.outputs[r] = self.ort[r].output_names
        self.extract_model_info_skl()
        self.extract_model_info_onnx(ort_size=len(content))

    def extract_model_info_skl(self, **kwargs):
        """
        Populates member ``self.skl_info`` with additional
        information on the model such as the number of node for
        a decision tree.
        """
        self.skl_info = dict(
            skl_nb_base_estimators=get_nb_skl_base_estimators(self.skl, fitted=True))
        self.skl_info.update(kwargs)
        if isinstance(self.skl, BaseDecisionTree):
            self.skl_info["skl_dt_nodes"] = self.skl.tree_.node_count
        elif isinstance(self.skl, BaseForest):
            self.skl_info["skl_rf_nodes"] = sum(
                est.tree_.node_count for est in self.skl.estimators_)

    def extract_model_info_onnx(self, **kwargs):
        """
        Populates member ``self.onnx_info`` with additional
        information on the :epkg:`ONNX` graph.
        """
        self.onnx_info = {
            'onnx_nodes': len(self.ort_onnx.graph.node),  # pylint: disable=E1101
            'onnx_opset': get_opset_number_from_onnx(),
        }
        self.onnx_info.update(kwargs)

    def data(self, N=None, dim=None, **kwargs):  # pylint: disable=W0221
        """
        Generates random features.

        @param      N       number of observations
        @param      dim     number of features
        """
        if dim is None:
            raise RuntimeError(  # pragma: no cover
                "dim must be defined.")
        if N is None:
            raise RuntimeError(  # pragma: no cover
                "N must be defined.")
        return self._get_random_dataset(N, dim)[:1]

    def model_info(self, model):
        """
        Returns additional informations about a model.

        @param      model       model to describe
        @return                 dictionary with additional descriptor
        """
        res = dict(type_name=model.__class__.__name__)
        return res

    def validate(self, results, **kwargs):
        """
        Checks that methods *predict* and *predict_proba* returns
        the same results for both :epkg:`scikit-learn` and
        :epkg:`onnxruntime`.
        """
        res = {}
        baseline = None
        for idt, fct, vals in results:
            key = idt, fct.get('method', '')
            if key not in res:
                res[key] = {}
            if isinstance(vals, list):
                vals = pandas.DataFrame(vals).values
            lib = fct['lib']
            res[key][lib] = vals
            if lib == 'skl':
                baseline = lib

        if len(res) == 0:
            raise RuntimeError(  # pragma: no cover
                "No results to compare.")
        if baseline is None:
            raise RuntimeError(  # pragma: no cover
                "Unable to guess the baseline in {}.".format(
                    list(res.pop())))

        for key, exp in res.items():
            vbase = exp[baseline]
            if vbase.shape[0] <= 10000:
                for name, vals in exp.items():
                    if name == baseline:
                        continue
                    p1, p2 = vbase, vals
                    if len(p1.shape) == 1 and len(p2.shape) == 2:
                        p2 = p2.ravel()
                    try:
                        assert_almost_equal(p1, p2, decimal=4)
                    except AssertionError as e:
                        msg = "ERROR: Dim {}-{} - discrepencies between '{}' and '{}' for '{}'.".format(
                            vbase.shape, vals.shape, baseline, name, key)
                        self.dump_error(msg, skl=self.skl, ort=self.ort,
                                        baseline=vbase, discrepencies=vals,
                                        onnx_bytes=self.ort_onnx.SerializeToString(),
                                        results=results, **kwargs)
                        raise AssertionError(msg) from e
