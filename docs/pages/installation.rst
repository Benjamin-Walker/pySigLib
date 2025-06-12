Installation
========================

Windows
--------

pySigLib requires an installation of the MSVC compiler in order to compile the package.
Please ensure this exists, then run:

.. code-block::

    pip install pysiglib

pySigLib will automatically detect CUDA, provided the `CUDA_PATH` environment variable is set correctly.
To manually disable CUDA and build pySigLib for CPU only, create an environment variable `CUSIG` and set
it to `0`:

.. code-block::

    set CUSIG=0
    pip install pysiglib

Similarly, the package will automatically detect if AVX2 instructions are supported.
To disable these manually, create an environment variable `CPSIG_AVX` and set it to `0`:

.. code-block::

    set CPSIG_AVX=0
    pip install pysiglib


Linux
-------

pySigLib requires an installation of the GCC compiler in order to compile the package.
Please ensure this exists, then run:

.. code-block::

    pip install pysiglib

pySigLib will automatically detect CUDA, provided the `CUDA_PATH` environment variable is set correctly.

Typically:

.. code-block::

    export CUDA_PATH=/usr/lib/nvidia-cuda-toolkit

To manually disable CUDA and build pySigLib for CPU only, create an environment variable `CUSIG` and set
it to `0`:

.. code-block::

    export CUSIG=0
    pip install pysiglib

Similarly, the package will automatically detect if AVX2 instructions are supported.
To disable these manually, create an environment variable `CPSIG_AVX` and set it to `0`:

.. code-block::

    export CPSIG_AVX=0
    pip install pysiglib

MacOS
-------

pySigLib requires an installation of the GCC compiler in order to compile the package.
Please ensure this exists, then run:

.. code-block::

    pip install pysiglib

pySigLib does not support CUDA or AVX2 instructions on MacOS, and will build without
them when installed.