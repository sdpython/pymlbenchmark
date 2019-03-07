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

    export LD_LIBRARY_PATH=/usr/local/Python-3.6.8
    export PYTHONPATH=export PYTHONPATH=~/onnxruntime/build/debian36/Release
    python3.6 ./onnxruntime/tools/ci_build/build.py --build_dir ./onnxruntime/build/debian36 --config Release --enable_pybind --build_wheel --use_mkldnn --use_openmp --build_shared_lib --numpy_version= --skip-keras-test

    export LD_LIBRARY_PATH=/usr/local/Python-3.7.2
    export PYTHONPATH=export PYTHONPATH=~/onnxruntime/build/debian37/Release
    python3.7 ./onnxruntime/tools/ci_build/build.py --build_dir ./onnxruntime/build/debian37 --config Release --enable_pybind --build_wheel --use_mkldnn --use_openmp --build_shared_lib --numpy_version= --skip-keras-test

If the wheel then, it is possible to just copy the files
into the *python* distribution:

::

    cp -r ./onnxruntime/build/debian36/Release/onnxruntime /usr/local/lib/python3.6/site-packages/
    cp -r ./onnxruntime/build/debian37/Release/onnxruntime /usr/local/lib/python3.7/site-packages/

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
