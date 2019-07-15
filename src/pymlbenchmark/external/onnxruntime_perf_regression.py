"""
@file
@brief Implements a benchmark for a single regression
about performance for :epkg:`onnxruntime`.
"""
import numpy
from ..datasets import random_regression
from .onnxruntime_perf import OnnxRuntimeBenchPerfTest


class OnnxRuntimeBenchPerfTestRegression(OnnxRuntimeBenchPerfTest):
    """
    Specific test to compare computing time predictions
    with :epkg:`scikit-learn` and :epkg:`onnxruntime`
    for a binary classification.
    See example :ref:`l-example-onnxruntime-linreg`.
    The class requires the following modules to be installed:
    :epkg:`onnx`, :epkg:`onnxruntime`, :epkg:`skl2onnx`,
    :epkg:`mlprodict`.
    """

    def _get_random_dataset(self, N, dim):
        """
        Returns a random datasets.
        """
        return random_regression(N, dim)

    def fcts(self, dim=None, **kwargs):  # pylint: disable=W0221
        """
        Returns a few functions, tests methods
        *perdict*, *predict_proba* for both
        :epkg:`scikit-learn` and :epkg:`OnnxInference`
        multiplied by the number of runtime to test.
        """
        def predict_skl_predict(X, model=self.skl):
            return model.predict(X)

        def predict_onnxrt_predict(X, sess, output):
            return numpy.array(sess.run({'X': X.astype(numpy.float32)})[output])

        fcts = [{'method': 'predict', 'lib': 'skl', 'fct': predict_skl_predict}]
        for runtime in self.ort:
            inst = self.ort[runtime]
            output = self.outputs[runtime][0]
            fcts.append({'method': 'predict', 'lib': 'onx' + runtime,
                         'fct': lambda X, sess=inst, output=output:
                         predict_onnxrt_predict(X, sess, output)})

        for fct in fcts:
            if fct['lib'] == 'skl':
                fct.update(self.skl_info)
            elif fct['lib'].startswith('onx'):
                fct.update(self.onnx_info)
        return fcts
