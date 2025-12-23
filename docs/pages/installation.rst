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


Linux
-------

pySigLib requires an installation of the GCC compiler in order to compile the package.
Please ensure this exists, then run:

.. code-block::

    pip install pysiglib

pySigLib will automatically detect CUDA, provided the `CUDA_PATH` environment variable is set correctly.
On most systems, this path will be ``/usr/lib/nvidia-cuda-toolkit`` and one can set it manually by running:

.. code-block::

    export CUDA_PATH=/usr/lib/nvidia-cuda-toolkit

To manually disable CUDA and build pySigLib for CPU only, create an environment variable `CUSIG` and set
it to `0`:

.. code-block::

    export CUSIG=0
    pip install pysiglib


MacOS
-------

pySigLib requires an installation of the GCC compiler in order to compile the package.
Please ensure this exists, then run:

.. code-block::

    pip install pysiglib

pySigLib does not support CUDA on MacOS, and will build without it when installed.
