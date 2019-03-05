"""
@file
@brief Helpers which returns more information about the system.
"""
import sys
import platform
from datetime import datetime
import numpy


def machine_information():
    """
    Returns information about the machine.

    @return     list of dictionaries

    .. runpython::
        :showcode:

        from pymlbenchmark.context import machine_information
        for row in machine_information():
            print(row)
    """
    return [
        {"name": "date", "version": str(datetime.now())},
        {"name": "numpy", "version": numpy.__version__},
        {"name": "python", "value": sys.version},
        {"name": "platform", "value": sys.platform},
        {"name": "OS", "value": platform.platform()},
        {"name": "machine", "value": platform.machine()},
        {"name": "processor", "value": platform.processor()},
        {"name": "release", "value": platform.release()},
        {"name": "architecture", "value": platform.architecture()},
    ]
