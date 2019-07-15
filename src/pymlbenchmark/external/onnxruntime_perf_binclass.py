"""
@file
@brief Implements a benchmark for a binary classification
about performance for :epkg:`OnnxInference`.
"""
import numpy
from ..datasets import random_binary_classification
from .onnxruntime_perf import OnnxRuntimeBenchPerfTest


class OnnxRuntimeBenchPerfTestBinaryClassification(OnnxRuntimeBenchPerfTest):
    """
    Specific test to compare computing time predictions
    with :epkg:`scikit-learn` and :epkg:`onnxruntime`
    for a binary classification.
    See example :ref:`l-example-onnxruntime-logreg`.
    The class requires the following modules to be installed:
    :epkg:`onnx`, :epkg:`onnxruntime`, :epkg:`skl2onnx`,
    :epkg:`mlprodict`.
    """

    def _get_random_dataset(self, N, dim):
        """
        Returns a random datasets.
        """
        return random_binary_classification(N, dim)

    def fcts(self, dim=None, **kwargs):  # pylint: disable=W0221
        """
        Returns a few functions, tests methods
        *perdict*, *predict_proba* for both
        :epkg:`scikit-learn` and :epkg:`OnnxInference`
        multiplied by the number of runtime to test.
        """
        def predict_skl_predict(X, model=self.skl):
            return model.predict(X)

        def predict_skl_predict_proba(X, model=self.skl):
            return model.predict_proba(X)

        def predict_onnxrt_predict(X, sess, output):
            return numpy.array(sess.run({'X': X.astype(numpy.float32)})[output])

        def predict_onnxrt_predict_proba(X, sess, output):
            res = sess.run({'X': X.astype(numpy.float32)})[output]
            # do not use DataFrame to convert the output into array,
            # it takes too much time
            out = numpy.empty((len(res), len(res[0])), dtype=numpy.float32)
            for i, row in enumerate(res):
                for k, v in row.items():
                    out[i, k] = v
            return out

        fcts = [{'method': 'predict', 'lib': 'skl', 'fct': predict_skl_predict}]
        for runtime in self.ort:
            inst = self.ort[runtime]
            output = self.outputs[runtime][0]
            fcts.append({'method': 'predict', 'lib': 'onx' + runtime,
                         'fct': lambda X, sess=inst, output=output:
                         predict_onnxrt_predict(X, sess, output)})

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
            fcts.append({'method': 'predict_proba', 'lib': 'skl',
                         'fct': predict_skl_predict_proba})
            for runtime in self.ort:
                inst = self.ort[runtime]
                output = self.outputs[runtime][1]
                fcts.append({'method': 'predict_proba', 'lib': 'onx' + runtime,
                             'fct': lambda X, sess=inst, output=output:
                             predict_onnxrt_predict_proba(X, sess, output)})

        for fct in fcts:
            if fct['lib'] == 'skl':
                fct.update(self.skl_info)
            elif fct['lib'].startswith('onx'):
                fct.update(self.onnx_info)
        return fcts
