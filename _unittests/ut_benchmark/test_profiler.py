# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import os
import unittest
from io import StringIO
import numpy
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from pymlbenchmark.benchmark import ProfilerCall


def to_array(li, dtype):
    return numpy.array(li, dtype=dtype)


def dot(a, b):
    return numpy.dot(a, b)


def fct1():
    return dot(to_array([0, 1, 2, 3, 4, 5], dtype=numpy.int32),
               to_array([1, 2, 3, 4, 5, 6], dtype=numpy.int32))


def fct2():
    return dot(to_array([0, 1, 2, 3, 4, 5], dtype=numpy.float32),
               to_array([1, 2, 3, 4, 5, 6], dtype=numpy.float32))


class TestProfiler(ExtTestCase):

    def test_profiler_call(self):
        prof = ProfilerCall()
        prof.profile(fct1, name="fct1")
        prof.profile(fct2, name="fct2")

        temp = get_temp_folder(__file__, "temp_profiler_call")
        dest = os.path.join(temp, "prof.txt")
        prof.to_txt(dest)

        with open(dest, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertTrue(prof.match(name="fct1"))
        self.assertIn("Duration", content)
        self.assertIn("to_array", content)
        self.assertIn("profiler_class", content)

    def test_profiler_call_filter(self):
        prof = ProfilerCall(fct_match=dict(name="fct1"))
        if prof.match(name='fct1'):
            prof.profile(fct1, name="fct1")
        if prof.match(name='fct2'):
            prof.profile(fct2, name="fct2")

        temp = get_temp_folder(__file__, "temp_profiler_call_filter")
        dest = os.path.join(temp, "prof.txt")
        prof.to_txt(dest)

        with open(dest, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertEqual(len(prof), 1)
        self.assertNotIn("fct2", content)

    def test_profiler_call_profile(self):
        prof = ProfilerCall(module="profile")
        prof.profile(fct2, name="fct2")
        st = StringIO()
        prof.to_txt(st)
        content = st.getvalue()
        self.assertEqual(len(prof), 1)
        self.assertNotIn('//', content)
        self.assertIn("fct2", content)

    def test_profiler_call_cprofile(self):
        prof = ProfilerCall(module="cProfile")
        prof.profile(fct2, name="fct2")
        st = StringIO()
        prof.to_txt(st)
        content = st.getvalue()
        self.assertEqual(len(prof), 1)
        self.assertNotIn('//', content)
        self.assertIn("fct2", content)


if __name__ == "__main__":
    unittest.main()
