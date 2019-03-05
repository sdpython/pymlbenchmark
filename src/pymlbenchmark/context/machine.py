"""
@file
@brief Helpers which returns more information about the system.
"""
import sys
import platform
from datetime import datetime


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
        import pandas
        print(pandas.DataFrame(machine_information(['numpy']):
    """
    res = [
        {"name": "date", "version": str(datetime.now())},
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
                res.append(
                    dict(name=name, version=getattr(mod, '__version__', None)))
            else:
                res.append(dict(name=name, version='not-imported'))
    return res
