"""
@file
@brief Shortcut to *external*.
"""

from .onnxruntime_perf_binclass import OnnxRuntimeBenchPerfTestBinaryClassification
from .onnxruntime_perf_regression import OnnxRuntimeBenchPerfTestRegression
from .onnxruntime_perf_list import (
    onnxruntime_perf_binary_classifiers,
    onnxruntime_perf_regressors,
    run_onnxruntime_test
)
