=====================
Benchmarked Libraries
=====================

Benchmarking is an exact science as the results
may change depending on the machine used to compute
the figures. There is not necessarily an exact correlation
between the processing time and the algorithm cost.
The results may also depend on the options used
to compile a library (CPU, GPU, MKL, ...).
Next sections gives some details on how it was done.

scikit-learn
============

:epkg:`scikit-learn` is usually the current latest
stable version except if the test involves a pull request
which implies :epkg:`scikit-learn` is installed from
the master branch.

onnxruntime
===========

:epkg:`onnxruntime` is not easy to install on Linux even on CPU.
The current implementation requires that :epkg:`Python` is built
with a specific flags ``--enable-shared``:

::

    ./configure --enable-optimizations --with-ensurepip=install --enable-shared --prefix=/opt/bin

This is due to a feature which requests to be able to interpret
*Python* inside a package itself and more specifically: `Embedding the Python interpreter
<https://pybind11.readthedocs.io/en/stable/compiling.html#embedding-the-python-interpreter>`_.
Then the environment variable ``LD_LIBRARY_PATH`` must be set to
the location of the shard libraries, ``/opt/bin`` in the previous example.
The following issue might appear:

::

    UserWarning: Cannot load onnxruntime.capi.
    Error: 'libnnvm_compiler.so: cannot open shared object file: No such file or directory'

To build :epkg:`onnxruntime`:

::

    git clone https://github.com/Microsoft/onnxruntime.git --recursive

    export LD_LIBRARY_PATH=/usr/local/Python-3.7.2
    export PYTHONPATH=/home/dupre/xadupre/onnxruntime/build/debian/Release
    python3.7 ./onnxruntime/tools/ci_build/build.py --build_dir ./onnxruntime/build/debian --config Release --build_wheel --use_mkldnn --use_openmp --use_llvm --numpy_version= --skip-keras-test

.. faqref::
    :title: cannot import name 'get_all_providers'

    The following error usually indicates than *onnxruntime* was
    compiled on one machine and used on another one with different
    dynamic libraries. Missing libraries needs to be installed
    or *onnxruntime* must be compiled on the machine it needs
    to be used.

    ::

        ImportError: cannot import name 'get_all_providers' from 'onnxruntime.capi._pybind_state'

Build mkl-dnn
=============

*onnxruntime* requires :epkg:`MKL-DNN`
(or *Math Kernel Library* for *Deep Neural Networks*)
if flags ``--use_mkldnn`` is used.
It can be built like the following:

::

    git clone https://github.com/intel/mkl-dnn.git
    cd scripts && ./prepare_mkl.sh && cd ..
    mkdir -p build && cd build && cmake $CMAKE_OPTIONS ..
    make
    ctest
    make install
