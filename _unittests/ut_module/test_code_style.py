"""
@brief      test log(time=0s)
"""
import os
import unittest
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import check_pep8, ExtTestCase


class TestCodeStyle(ExtTestCase):
    """Test style."""

    def test_style_src(self):
        thi = os.path.abspath(os.path.dirname(__file__))
        src_ = os.path.normpath(os.path.join(thi, "..", "..", "src"))
        check_pep8(src_, fLOG=fLOG,
                   pylint_ignore=('C0103', 'C1801', 'R0201', 'R1705', 'W0108', 'W0613',
                                  'R1702', 'W0212', 'W0640', 'W0223', 'W0201',
                                  'W0622', 'C0123', 'W0107', 'E0012', 'C0209'),
                   skip=["Instance of 'tuple' has no ",
                         "do not compare types, use 'isinstance()'",
                         "Instance of 'AutoAction' has no 'children' member",
                         "gactions.py:225: R1711",
                         "gactions.py:238: E1128",
                         "R1720",
                         "R0914",
                         ])

    def test_style_test(self):
        thi = os.path.abspath(os.path.dirname(__file__))
        test = os.path.normpath(os.path.join(thi, "..", ))
        check_pep8(test, fLOG=fLOG, neg_pattern="temp_.*",
                   pylint_ignore=('C0103', 'C1801', 'R0201', 'R1705', 'W0108', 'W0613',
                                  'C0111', 'W0107', 'C0415', 'E0012', 'C0209'),
                   skip=["Instance of 'tuple' has no ",
                         "R1720",
                         "R0914",
                         ])


if __name__ == "__main__":
    unittest.main()
