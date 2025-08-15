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
from ctypes import c_double, POINTER, cast

import numpy as np
import torch

from .transform_path import transform_path
from .load_siglib import CPSIG, CUSIG, BUILT_WITH_CUDA
from .param_checks import check_type
from .error_codes import err_msg
from .data_handlers import DoublePathInputHandler, ScalarOutputHandler, GridOutputHandler


def sig_kernel_(data, result, gram, dyadic_order_1, dyadic_order_2, n_jobs, return_grid):

    err_code = CPSIG.batch_sig_kernel(
        cast(gram.data_ptr(), POINTER(c_double)),
        result.data_ptr,
        data.batch_size,
        data.dimension,
        data.length_1,
        data.length_2,
        dyadic_order_1,
        dyadic_order_2,
        n_jobs,
        return_grid
    )

    if err_code:
        raise Exception("Error in pysiglib.sig_kernel: " + err_msg(err_code))

def sig_kernel_cuda_(data, result, gram, dyadic_order_1, dyadic_order_2, return_grid):
    err_code = CUSIG.batch_sig_kernel_cuda(
        cast(gram.data_ptr(), POINTER(c_double)),
        result.data_ptr, data.batch_size,
        data.dimension,
        data.length_1,
        data.length_2,
        dyadic_order_1,
        dyadic_order_2,
        return_grid
    )

    if err_code:
        raise Exception("Error in pysiglib.sig_kernel: " + err_msg(err_code))

def sig_kernel(
        path1 : Union[np.ndarray, torch.tensor],
        path2 : Union[np.ndarray, torch.tensor],
        dyadic_order : Union[int, tuple],
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        n_jobs : int = 1,
        return_grid = False
) -> Union[np.ndarray, torch.tensor]:
    """
    Computes a single signature kernel or a batch of signature kernels.
    The signature kernel of two :math:`d`-dimensional paths :math:`x,y`
    is defined as

    .. math::

        k_{x,y}(s,t) := \\left< S(x)_{[0,s]}, S(y)_{[0, t]} \\right>_{T((\\mathbb{R}^d))}

    where the inner product is defined as

    .. math::

        \\left< A, B \\right> := \\sum_{k=0}^{\\infty} \\left< A_k, B_k \\right>_{\\left(\\mathbb{R}^d\\right)^{\\otimes k}}
    .. math::

        \\left< u, v \\right>_{\\left(\\mathbb{R}^d\\right)^{\\otimes k}} := \\prod_{i=1}^k \\left< u_i, v_i \\right>_{\\mathbb{R}^d}

    :param path1: The first underlying path or batch of paths, given as a `numpy.ndarray` or
        `torch.tensor`. For a single path, this must be of shape (length, dimension). For a
        batch of paths, this must be of shape (batch size, length, dimension).
    :type path1: numpy.ndarray | torch.tensor
    :param path2: The second underlying path or batch of paths, given as a `numpy.ndarray`
        or `torch.tensor`. For a single path, this must be of shape (length, dimension).
        For a batch of paths, this must be of shape (batch size, length, dimension).
    :type path2: numpy.ndarray | torch.tensor
    :param dyadic_order: If set to a positive integer :math:`\\lambda`, will refine the
        PDE grid by a factor of :math:`2^\\lambda`.
    :type dyadic_order: int | tuple
    :param time_aug: If set to True, will compute the signature of the time-augmented path, :math:`\\hat{x}_t := (t, x_t)`,
        defined as the original path with an extra channel set to time, :math:`t`. This channel spans :math:`[0, t_L]`,
        where :math`t_L` is given by the parameter ``end_time``.
    :type time_aug: bool
    :param lead_lag: If set to True, will compute the signature of the path after applying the lead-lag transformation.
    :type lead_lag: bool
    :param end_time: End time for time-augmentation, :math:`t_L`.
    :type end_time: float
    :param n_jobs: (Only applicable to CPU computation) Number of threads to run in parallel.
        If n_jobs = 1, the computation is run serially. If set to -1, all available threads
        are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example
        if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :param return_grid: If ``True``, returns the entire PDE grid.
    :type return_grid: bool
    :return: Single signature kernel or batch of signature kernels
    :rtype: numpy.ndarray | torch.tensor

    .. note::

        Ideally, any array passed to ``pysiglib.sig_kernel`` should be both contiguous and own its data.
        If this is not the case, ``pysiglib.sig_kernel`` will internally create a contiguous copy, which may be
        inefficient.
    """
    check_type(n_jobs, "n_jobs", int)
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")

    if isinstance(dyadic_order, tuple) and len(dyadic_order) == 2:
        dyadic_order_1 = dyadic_order[0]
        dyadic_order_2 = dyadic_order[1]
    elif isinstance(dyadic_order, int):
        dyadic_order_1 = dyadic_order
        dyadic_order_2 = dyadic_order
    else:
        raise TypeError("dyadic_order must be an integer or a tuple of length 2")

    if dyadic_order_1 < 0 or dyadic_order_2 < 0:
        raise ValueError("dyadic_order must be a non-negative integer or tuple of non-negative integers")

    if time_aug or lead_lag:
        path1 = transform_path(path1, time_aug, lead_lag, end_time, n_jobs)
        path2 = transform_path(path2, time_aug, lead_lag, end_time, n_jobs)

    data = DoublePathInputHandler(path1, path2, False, False, 0., "path1", "path2")

    if not return_grid:
        result = ScalarOutputHandler(data)
    else:
        dyadic_len_1 = ((data.length_1 - 1) << dyadic_order_1) + 1
        dyadic_len_2 = ((data.length_2 - 1) << dyadic_order_2) + 1
        result = GridOutputHandler(dyadic_len_1, dyadic_len_2, data)

    torch_path1 = torch.as_tensor(data.path1, dtype = torch.double)  # Avoids data copy
    torch_path2 = torch.as_tensor(data.path2, dtype = torch.double)

    if data.is_batch:
        x1 = torch_path1[:, 1:, :] - torch_path1[:, :-1, :]
        y1 = torch_path2[:, 1:, :] - torch_path2[:, :-1, :]
    else:
        x1 = (torch_path1[1:, :] - torch_path1[:-1, :])[None, :, :]
        y1 = (torch_path2[1:, :] - torch_path2[:-1, :])[None, :, :]

    gram = torch.empty((x1.shape[0], x1.shape[1], y1.shape[1]), dtype=torch.double, device = x1.device)
    torch.bmm(x1, y1.permute(0, 2, 1), out=gram)

    if data.device == "cpu":
        sig_kernel_(data, result, gram, dyadic_order_1, dyadic_order_2, n_jobs, return_grid)
    else:
        if not BUILT_WITH_CUDA:
            raise RuntimeError("pySigLib was build without CUDA - data must be moved to CPU.")
        sig_kernel_cuda_(data, result, gram, dyadic_order_1, dyadic_order_2, return_grid)

    return result.data
