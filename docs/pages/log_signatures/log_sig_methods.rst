Computing Log Signatures
===========================

For :math:`x \in T(\mathbb{R}^d)`, the logarithm in tensor space is defined by

.. math::

    \log(1 + x) = \sum_{n \geq 1} \frac{(-1)^{n-1} x^n}{n}.

Applying this logarithm map to the signature yields the `log signature`. A
useful property of the log signature is its ability to be `compressed` into
a smaller tensor without losing information, by considering its values at
`Lyndon words`. ``pysiglib`` implements three methods for
log signature computation, controlled by the ``method`` flag, described in
detail below.

Preparing for Log Signature Computations
------------------------------------------

Before computing the log signature, ``pysiglib`` requires a call to
``pysiglib.prepare_log_sig``, which will pre-compute and cache certain
objects which are required for the computation. This function should be
run only once before the computation, for each required ``(dimension, degree, method)``
combination. This call is not thread safe.

.. code-block:: python

    import torch
    import pysiglib

    # We know in advance that we will need log signatures
    # for dimensions 5 and 10 with degree 3 and method 2.
    pysiglib.prepare_log_sig(5, 3, method=2)
    pysiglib.prepare_log_sig(10, 3, method=2)

    for i in range(10):
        X = torch.rand((200, 5))
        Y = torch.rand((100, 10))

        X_ls = pysiglib.log_sig(X, 3, method=2)
        Y_ls = pysiglib.log_sig(Y, 3, method=2)

The ordering of the methods is chosen such that higher
methods are strictly more complicated than lower methods.
As such, preparing for ``method=2`` is also sufficient to
run ``method=1``. We note also that ``method=0`` does not
require a call to ``pysiglib.prepare_log_sig``.

.. code-block:: python

    import torch
    import pysiglib

    X = torch.rand((100, 5))

    pysiglib.log_sig(X, 3, method=0) # No error: method=0 does not require preparation

    pysiglib.prepare_log_sig(5, 3, method=2)
    pysiglib.log_sig(X, 3, method=1) # No error: prepare already called with a higher method

Methods
--------

We give a brief overview of the methods below. For details concerning the Lyndon
basis, we refer the user to the documentation for the ``iisignature`` and
``signatory`` packages `here <https://github.com/bottler/phd-docs/blob/master/iisignature.pdf>`__
and `here <https://signatory.readthedocs.io/en/latest/>`__, as well as their corresponding papers
`here <https://arxiv.org/pdf/1802.08252>`__ and `here <https://arxiv.org/pdf/2001.00706>`__.

``method = 0``
-------------------

This option corresponds to the full uncompressed log signature, obtained by first
computing the signature and then applying the tensor logarithm. The output is the
same length as a signature.

This method corresponds to ``methods="x"`` in the ``iisignature`` package and
``mode="expand"`` in the ``signatory`` package.

``method = 1``
-------------------

This option computes the log signature as in the ``method=0`` case, and then extracts
those coefficients which are indexed by `Lyndon words`. Whilst the result is not
strictly the log signature, it is equivalent to the log signature up to a linear
transformation. This makes it a faster alternative to ``method=2``, suitable
for machine learning applications where the missing linear transformation
can be learnt implicitly.

This method corresponds to ``mode="words"`` in the ``signatory`` package.

``method = 2``
-------------------

This option computes the log signature as in the ``method=2`` case, but
projecting the result to the `Lyndon basis`. The result is the compressed
log signature. This method is required for use cases outside of machine
learning, where the Lyndon basis projection cannot be omitted as in the
``method=1`` case.

This method corresponds to ``methods="s"`` in the ``iisignature`` package and
``mode="brackets"`` in the ``signatory`` package.
