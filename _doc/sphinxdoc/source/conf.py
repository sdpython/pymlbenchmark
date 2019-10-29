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
    'css_files': get_default_stylesheet() + ['_static/my-styles.css', '_static/gallery.css'],
}

html_logo = "project_ico.png"

if html_theme == "bootstrap":
    html_theme_options = {
        'navbar_title': "BASE",
        'navbar_site_name': "Site",
        'navbar_links': [
            ("XD", "http://www.xavierdupre.fr", True),
        ],
        'navbar_sidebarrel': True,
        'navbar_pagenav': True,
        'navbar_pagenav_name': "Page",
        'bootswatch_theme': "readable",
        # united = weird colors, sandstone=green, simplex=red, paper=trop bleu
        # lumen: OK
        # to try, yeti, flatly, paper
        'bootstrap_version': "3",
        'source_link_position': "footer",
    }

language = "en"

mathdef_link_only = True

epkg_dictionary.update({
    'Benchmarks about Machine Learning': 'http://www.xavierdupre.fr/app/_benchmarks/helpsphinx/index.html',
    'C': "https://en.wikipedia.org/wiki/C_(programming_language)",
    'cffi': "https://cffi.readthedocs.io/en/latest/",
    'cProfile': 'https://docs.python.org/3/library/profile.html#module-cProfile',
    'DataFrame': 'https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html',
    'MKL-DNN': 'https://github.com/intel/mkl-dnn',
    'mlprodict': 'http://www.xavierdupre.fr/app/mlprodict/helpsphinx/index.html',
    'onnx': 'https://github.com/onnx/onnx',
    'ONNX': 'https://onnx.ai/',
    'OnnxInference': 'http://www.xavierdupre.fr/app/mlprodict/helpsphinx/mlprodict/onnxrt/onnx_inference.html',
    'onnxruntime': 'https://github.com/Microsoft/onnxruntime',
    'PolynomialFeatures': 'https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html',
    'pyinstrument': 'https://github.com/joerick/pyinstrument',
    'pymlbenchmark': blog_root,
    'Python': 'https://www.python.org/',
    'skl2onnx': 'https://github.com/onnx/sklearn-onnx',
    'tqdm': 'https://github.com/tqdm/tqdm',
    'scikit-learn': 'https://scikit-learn.org/stable/',
})
