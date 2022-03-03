"""
@file
@brief Helpers which returns more information about the system.
"""
import sys
import io
import platform
import contextlib
from datetime import datetime
import numpy
from cpuinfo import get_cpu_info  # pylint: disable=E0401


def get_numpy_info():
    """
    Retrieves information about numpy compilation.
    """
    s = io.StringIO()
    with contextlib.redirect_stdout(s):
        numpy.show_config()
    return s.getvalue()


def machine_information(pkgs=None):
    """
    Returns information about the machine.

    @param      pkgs    if not None, adds version information for the
                        listed packages
    @return             list of dictionaries

    .. runpython::
        :showcode:

        from pymlbenchmark.context import machine_information
        for row in machine_information():
            print(row)

    .. runpython::
        :showcode:

        from pymlbenchmark.context import machine_information
        import numpy
        import pandas
        print(pandas.DataFrame(machine_information(['numpy'])))
    """
    res = [
        {"name": "date", "version": str(datetime.now()).split()[0]},
        {"name": "python", "value": sys.version},
        {"name": "platform", "value": sys.platform},
        {"name": "OS", "value": platform.platform()},
        {"name": "machine", "value": platform.machine()},
        {"name": "processor", "value": platform.processor()},
        {"name": "release", "value": platform.release()},
        {"name": "architecture", "value": platform.architecture()},
    ]
    info = get_cpu_info()
    for k, v in sorted(info.items()):
        if k in {'brand_raw', 'count', 'arch', 'processor_type',
                 'hz_advertised', 'stepping',
                 'l1_cache_size', 'l2_cache_size',
                 'l3_cache_size', 'l1_data_cache_size',
                 'l1_instruction_cache_size', 'l2_cache_line_size',
                 'l2_cache_associativity'}:
            res.append(dict(name=k, value=v))
        if k == 'flags':
            res.append(dict(name=k, value=' '.join(v)))
    if pkgs is not None:
        for name in sorted(pkgs):
            if name in sys.modules:
                mod = sys.modules[name]
                obs = dict(name=name, version=getattr(
                    mod, '__version__', 'not-found'))
                if name == "onnxruntime":
                    obs['value'] = mod.get_device()
                elif name == 'numpy':
                    sinfo = get_numpy_info()
                    info = []
                    for sub in ['mkl_lapack95_lp64', 'mkl_blas95_lp64', 'openblas',
                                "language = c"]:
                        if sub in sinfo:
                            info.append(sub.replace(' ', ''))
                    obs['value'] = ", ".join(info)
                elif name == "onnx":
                    from mlprodict import __max_supported_opset_  # pylint: disable=C0415
                    obs['value'] = "opset={}".format(
                        __max_supported_opset_)
                res.append(obs)
            else:
                res.append(dict(name=name, version='not-imported'))
    return res
