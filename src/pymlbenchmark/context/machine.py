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
                    from mlprodict.tools.asv_options_helper import benchmark_version
                    obs['value'] = "opset={}/{}".format(
                        onnx_opset_version(),
                        benchmark_version()[-1])
                res.append(obs)
            else:
                res.append(dict(name=name, version='not-imported'))
    return res
