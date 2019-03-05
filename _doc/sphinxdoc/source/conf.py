# -*- coding: utf-8 -*-
import sys
import os
import sphinx_bootstrap_theme
from pyquickhelper.helpgen.default_conf import set_sphinx_variables, get_default_stylesheet


sys.path.insert(0, os.path.abspath(os.path.join(os.path.split(__file__)[0])))

local_template = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), "phdoc_templates")

set_sphinx_variables(__file__, "pymlbenchmark", "Xavier Dupré", 2019,
                     "bootstrap", sphinx_bootstrap_theme.get_html_theme_path(),
                     locals(), extlinks=dict(
                         issue=('https://github.com/sdpython/pymlbenchmark/issues/%s', 'issue')),
                     title="Benchmarks around Machine Learning with Python", book=False)

blog_root = "http://www.xavierdupre.fr/app/pymlbenchmark/helpsphinx/"

html_context = {
    'css_files': get_default_stylesheet() + ['_static/my-styles.css'],
}

html_logo = "project_ico.png"

html_sidebars = {}

language = "en"

mathdef_link_only = True

epkg_dictionary.update({
    'C': "https://en.wikipedia.org/wiki/C_(programming_language)",
    'cffi': "https://cffi.readthedocs.io/en/latest/",
    'onnx': 'https://github.com/onnx/onnx',
    'ONNX': 'https://onnx.ai/',
    'onnxruntime': 'https://github.com/Microsoft/onnxruntime',
    'PolynomialFeatures': 'https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html',
    'Python': 'https://www.python.org/',
    'scikit-learn': 'https://scikit-learn.org/stable/',
})
