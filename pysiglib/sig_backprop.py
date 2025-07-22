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
from ctypes import c_double, POINTER

import numpy as np
import torch

from .param_checks import check_cpu, check_type
from .load_siglib import CPSIG
from .error_codes import err_msg
from .data_handlers import PathInputHandler, SigOutputHandler, PathOutputHandler, SigInputHandler, DoubleSigInputHandler, TripleSigInputHandler
from .dtypes import CPSIG_SIG_BACKPROP, CPSIG_BATCH_SIG_BACKPROP
from .sig_length import sig_length

def sig_combine_backprop_(sig_data, sig1_deriv, sig2_deriv, dimension, degree):
    err_code = CPSIG.sig_combine_backprop(
        sig_data.sig3_ptr,
        sig1_deriv.data_ptr,
        sig2_deriv.data_ptr,
        sig_data.sig1_ptr,
        sig_data.sig2_ptr,
        dimension,
        degree
    )

    if err_code:
        raise Exception("Error in pysiglib.sig_combine_backprop: " + err_msg(err_code))
    return sig1_deriv.data, sig2_deriv.data

def batch_sig_combine_backprop_(sig_data, sig1_deriv, sig2_deriv, dimension, degree, n_jobs):
    err_code = CPSIG.batch_sig_combine_backprop(
        sig_data.sig3_ptr,
        sig1_deriv.data_ptr,
        sig2_deriv.data_ptr,
        sig_data.sig1_ptr,
        sig_data.sig2_ptr,
        sig_data.batch_size,
        dimension,
        degree,
        n_jobs
    )

    if err_code:
        raise Exception("Error in pysiglib.sig_combine_backprop: " + err_msg(err_code))
    return sig1_deriv.data, sig2_deriv.data

def sig_combine_backprop(
        sig_combined_deriv : Union[np.ndarray, torch.tensor],
        sig1 : Union[np.ndarray, torch.tensor],
        sig2 : Union[np.ndarray, torch.tensor],
        dimension : int,
        degree : int,
        n_jobs : int = 1
):
    """
    This function is required to backpropogate through ``pysiglib.sig_combine``.
    Given the derivatives of a scalar function :math:`F` with respect to the
    result of ``pysiglib.sig_combine``, :math:`\\partial F / \\partial S(x_1 * x_2)`,
    returns the derivatives of :math:`F` with respect to the original two signatures,
    :math:`\\partial F / \\partial S(x_1)` and :math:`\\partial F / \\partial S(x_2)`.

    :param sig_combined_deriv: Derivative with respect to the combined signature,
        :math:`\\partial F / \\partial S(x_1 * x_2)`
    :type sig_combine_deriv: numpy.ndarray | torch.tensor
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
    :return: Derivatives with respect to sig1 and sig2
    :rtype: Tuple[numpy.ndarray | torch.tensor, numpy.ndarray | torch.tensor]

    .. note::

        Ideally, any array passed to ``pysiglib.sig_combine_backprop`` should be both contiguous and own its data.
        If this is not the case, ``pysiglib.sig_combine_backprop`` will internally create a contiguous copy, which may be
        inefficient.

    """

    check_cpu(sig_combined_deriv, "sig_combined_deriv")
    check_cpu(sig1, "sig1")
    check_cpu(sig2, "sig2")

    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)

    sig_len = sig_length(dimension, degree)
    sig_data = TripleSigInputHandler(sig1, sig2, sig_combined_deriv, sig_len, "sig1", "sig2", "sig_combined_deriv")

    sig1_deriv = SigOutputHandler(sig_data, sig_len)
    sig2_deriv = SigOutputHandler(sig_data, sig_len)

    if sig_data.is_batch:
        check_type(n_jobs, "n_jobs", int)
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")
        return batch_sig_combine_backprop_(sig_data, sig1_deriv, sig2_deriv, dimension, degree, n_jobs)
    return sig_combine_backprop_(sig_data, sig1_deriv, sig2_deriv, dimension, degree)

def sig_backprop_(path_data, sig_data, result, degree, time_aug, lead_lag):
    err_code = CPSIG_SIG_BACKPROP[path_data.dtype](
        path_data.data_ptr,
        result.data_ptr,
        sig_data.sig2_ptr,
        sig_data.sig1_ptr,
        path_data.data_dimension,
        path_data.data_length,
        degree,
        time_aug,
        lead_lag
    )

    if err_code:
        raise Exception("Error in pysiglib.sig_backprop: " + err_msg(err_code))
    return result.data

def batch_sig_backprop_(path_data, sig_data, result, degree, time_aug, lead_lag, n_jobs):
    err_code = CPSIG_BATCH_SIG_BACKPROP[path_data.dtype](
        path_data.data_ptr,
        result.data_ptr,
        sig_data.sig2_ptr,
        sig_data.sig1_ptr,
        path_data.batch_size,
        path_data.data_dimension,
        path_data.data_length,
        degree,
        time_aug,
        lead_lag,
        n_jobs
    )

    if err_code:
        raise Exception("Error in pysiglib.sig_backprop: " + err_msg(err_code))
    return result.data

def sig_backprop(
        path : Union[np.ndarray, torch.tensor],
        sig : Union[np.ndarray, torch.tensor],
        sig_derivs : Union[np.ndarray, torch.tensor],
        degree : int,
        time_aug : bool = False,
        lead_lag : bool = False,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    """
    This function is required to backpropogate through the signature computation.
    Given the derivatives of a scalar function :math:`F` with respect to the
    signature, :math:`\\partial F / \\partial S(x)`, returns the
    derivatives of :math:`F` with respect to the underlying path,
    :math:`\\partial F / \\partial x`.

    :param path: The underlying path or batch of paths, given as a `numpy.ndarray` or `torch.tensor`.
        For a single path, this must be of shape (length, dimension). For a batch of paths, this must
        be of shape (batch size, length, dimension).
    :type path: numpy.ndarray | torch.tensor
    :param sig: Signature(s) of the path or batch of paths.
    :type sig: numpy.ndarray | torch.tensor
    :param sig_derivs: Derivatives of the scalar function :math:`F` with respect to the signature(s),
        :math:`\\partial F / \\partial S(x)`. This must be an array of the same shape as the
        provided signature(s).
    :type sig_derivs: numpy.ndarray | torch.tensor
    :param degree: The truncation level of the signature, :math:`N`.
    :type degree: int
    :param time_aug: If set to True, will compute the signature of the time-augmented path, :math:`\\hat{x}_t := (t, x_t)`,
        defined as the original path with an extra channel set to time, :math:`t`.
    :type time_aug: bool
    :param lead_lag: If set to True, will compute the signature of the path after applying the lead-lag transformation.
    :type lead_lag: bool
    :return: Derivatives of the scalar function :math:`F` with respect to the path(s), :math:`\\partial F / \\partial x`.
        This is an array of the same shape as the provided path(s).
    :rtype: numpy.ndarray | torch.tensor


    .. note::

        Ideally, any array passed to ``pysiglib.sig_backprop`` should be both contiguous and own its data.
        If this is not the case, ``pysiglib.sig_backprop`` will internally create a contiguous copy, which may be
        inefficient.

    """

    check_cpu(path, "path")
    check_cpu(sig, "sig")
    check_cpu(sig_derivs, "sig_derivs")
    path_data = PathInputHandler(path, time_aug, lead_lag, "path")
    sig_len = sig_length(path_data.dimension, degree)
    sig_data = DoubleSigInputHandler(sig, sig_derivs, sig_len, "sig", "sig_derivs")
    result = PathOutputHandler(path_data.data_length, path_data.data_dimension, path_data.batch_size, path_data.is_batch, path_data.type_, path_data.device)

    if path_data.is_batch != sig_data.is_batch or path_data.batch_size != sig_data.batch_size:
        raise ValueError("path, sig and sig_derivs must have the same batch sizes")

    if path_data.is_batch:
        check_type(n_jobs, "n_jobs", int)
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")
        return batch_sig_backprop_(path_data, sig_data, result, degree, time_aug, lead_lag, n_jobs)
    return sig_backprop_(path_data, sig_data, result, degree, time_aug, lead_lag)