# Copyright 2025 Daniil Shmelev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from typing import Union

import numpy as np
import torch

from .load_siglib import CPSIG
from .param_checks import check_type, check_non_neg, check_cpu
from .error_codes import err_msg
from .dtypes import CPSIG_SIGNATURE, CPSIG_BATCH_SIGNATURE
from .sig_length import sig_length


######################################################
# Python wrappers
######################################################

from .data_handlers import PathInputHandler, DoubleSigInputHandler, SigOutputHandler


def sig_combine(
        sig1 : Union[np.ndarray, torch.tensor],
        sig2 : Union[np.ndarray, torch.tensor],
        dimension : int,
        degree : int,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    """
    Combines two truncated signatures of the same degree and dimension into one signature. In particular, let :math:`x_1, x_2`
    be two paths such that the first point of :math:`x_2` is the last point of :math:`x_1`. Let :math:`S(x_1), S(x_2)`
    be the truncated signatures of :math:`x_1, x_2` respectively. Then calling this function on :math:`S(x_1), S(x_2)` returns
    the truncated signature of the concatenated path,

    .. math::

        S(x_1 * x_2) = S(x_1) \\otimes S(x_2),

    where :math:`x_1 * x_2` is the concatenation of the two paths :math:`x_1, x_2`.

    :param sig1: The first truncated signature
    :type sig1: numpy.ndarray | torch.tensor
    :param sig2: The second truncated signature. Must have the same degree and dimension as the first.
    :type sig2: numpy.ndarray | torch.tensor
    :param dimension: Dimension of the underlying space, :math:`d`.
    :type dimension: int
    :param degree: Truncation level of the signatures, :math:`N`
    :type degree: int
    :param n_jobs: Number of threads to run in parallel. If n_jobs = 1, the computation is run serially.
        If set to -1, all available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs)
        threads are used. For example if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :return: Combined signature, :math:`S(x_1 * x_2)`
    :rtype: numpy.ndarray | torch.tensor

    .. note::

        Parallelising the computation by setting ``n_jobs != 1`` can be beneficial when the
        workload is large. However, if the workload is too small, it may be faster to set this
        to ``1`` and run the computation serially, due to parallelisation overhead.

    .. note::

        Ideally, any array passed to ``pysiglib.sig_combine`` should be both contiguous and own its data.
        If this is not the case, ``pysiglib.sig_combine`` will internally create a contiguous copy, which may be
        inefficient.

    Example usage::

        import pysiglib

        batch_size = 32
        length = 100
        dimension = 5
        degree = 3

        X1 = np.random.uniform(size=(batch_size, length, dimension))
        X2 = np.random.uniform(size=(batch_size, length, dimension))
        X_concat = np.concatenate((X1, X2), axis=1)

        X2 = np.concatenate((X1[:, [-1], :], X2), axis=1) # Make sure first pt of X2 is last pt of X1
        sig1 = pysiglib.signature(X1, degree)
        sig2 = pysiglib.signature(X2, degree)

        # The tensor product...
        sig_mult = pysiglib.sig_combine(sig1, sig2, dimension, degree)

        # ... is the same as the signature of the concatenated path:
        sig = pysiglib.signature(X_concat, degree)
    """

    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)
    check_non_neg(dimension, "dimension")
    check_non_neg(degree, "degree")
    check_type(n_jobs, "n_jobs", int)
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")

    check_cpu(sig1, "sig1")
    check_cpu(sig2, "sig2")

    sig_len = sig_length(dimension, degree)
    data = DoubleSigInputHandler(sig1, sig2, sig_len, "sig1", "sig2")
    result = SigOutputHandler(data, sig_len)

    if data.is_batch:
        err_code = CPSIG.batch_sig_combine(
            data.sig1_ptr,
            data.sig2_ptr,
            result.data_ptr,
            data.batch_size,
            dimension,
            degree,
            n_jobs
        )
    else:
        err_code = CPSIG.sig_combine(
            data.sig1_ptr,
            data.sig2_ptr,
            result.data_ptr,
            dimension,
            degree
        )

    if err_code:
        raise Exception("Error in pysiglib.signature: " + err_msg(err_code))
    return result.data

def signature_(data, result, degree, horner = True):
    err_code = CPSIG_SIGNATURE[data.dtype](
        data.data_ptr,
        result.data_ptr,
        data.data_dimension,
        data.data_length,
        degree,
        data.time_aug,
        data.lead_lag,
        data.end_time,
        horner
    )

    if err_code:
        raise Exception("Error in pysiglib.signature: " + err_msg(err_code))
    return result.data

def batch_signature_(data, result, degree, horner = True, n_jobs = 1):
    err_code = CPSIG_BATCH_SIGNATURE[data.dtype](
        data.data_ptr,
        result.data_ptr,
        data.batch_size,
        data.data_dimension,
        data.data_length,
        degree,
        data.time_aug,
        data.lead_lag,
        data.end_time,
        horner,
        n_jobs
    )

    if err_code:
        raise Exception("Error in pysiglib.signature: " + err_msg(err_code))
    return result.data

def signature(
        path : Union[np.ndarray, torch.tensor],
        degree : int,
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        horner : bool = True,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    """
    Computes the truncated signature of single path or a batch of paths. For
    a single path :math:`x`, the signature is given by

    .. math::

        S(x)_{[s,t]} := \\left( 1, S(x)^{(1)}_{[s,t]}, \\ldots, S(x)^{(N)}_{[s,t]}\\right) \\in T((\\mathbb{R}^d)),
    .. math::

        S(x)^{(k)}_{[s,t]} := \\int_{s < t_1 < \\cdots < t_k < t} dx_{t_1} \\otimes dx_{t_2} \\otimes \\cdots \\otimes dx_{t_k} \\in \\left(\\mathbb{R}^d\\right)^{\\otimes k}.

    :param path: The underlying path or batch of paths, given as a `numpy.ndarray` or `torch.tensor`.
        For a single path, this must be of shape (length, dimension). For a batch of paths, this must
        be of shape (batch size, length, dimension).
    :type path: numpy.ndarray | torch.tensor
    :param degree: The truncation level of the signature, :math:`N`.
    :type degree: int
    :param time_aug: If set to True, will compute the signature of the time-augmented path, :math:`\\hat{x}_t := (t, x_t)`,
        defined as the original path with an extra channel set to time, :math:`t`. This channel spans :math:`[0, t_L]`,
        where :math`t_L` is given by the parameter ``end_time``.
    :type time_aug: bool
    :param lead_lag: If set to True, will compute the signature of the path after applying the lead-lag transformation.
    :type lead_lag: bool
    :param end_time: End time for time-augmentation, :math:`t_L`.
    :type end_time: float
    :param horner: If True, will use Horner's algorithm for polynomial multiplication.
    :type horner: bool
    :param n_jobs: Number of threads to run in parallel. If n_jobs = 1, the computation is run serially.
        If set to -1, all available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs)
        threads are used. For example if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :return: Truncated signature, or a batch of truncated signatures.
    :rtype: numpy.ndarray | torch.tensor

    .. note::

        Ideally, any array passed to ``pysiglib.signature`` should be both contiguous and own its data.
        If this is not the case, ``pysiglib.signature`` will internally create a contiguous copy, which may be
        inefficient.

    """
    check_type(degree, "degree", int)
    check_type(horner, "horner", bool)

    check_cpu(path, "path")

    data = PathInputHandler(path, time_aug, lead_lag, end_time, "path")
    sig_len = sig_length(data.dimension, degree)
    result = SigOutputHandler(data, sig_len)
    if data.is_batch:
        check_type(n_jobs, "n_jobs", int)
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")
        return batch_signature_(data, result, degree, horner, n_jobs)
    return signature_(data, result, degree, horner)
