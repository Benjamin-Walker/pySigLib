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

from typing import Union, Tuple
from ctypes import c_double, POINTER, cast

import numpy as np
import torch

from .transform_path import transform_path
from .transform_path_backprop import transform_path_backprop
from .sig_kernel import sig_kernel
from .load_siglib import CPSIG, CUSIG, BUILT_WITH_CUDA
from .param_checks import check_type
from .error_codes import err_msg
from .data_handlers import DoublePathInputHandler, ScalarInputHandler, ScalarOutputHandler, GridOutputHandler, PathOutputHandler, PathInputHandler

def sig_kernel_backprop_(data, derivs_data, result, gram, k_grid_data, dyadic_order_1, dyadic_order_2, n_jobs):

    err_code = CPSIG.batch_sig_kernel_backprop(
        cast(gram.data_ptr(), POINTER(c_double)),
        result.data_ptr,
        derivs_data.data_ptr,
        k_grid_data.data_ptr,
        data.batch_size,
        data.dimension,
        data.length_1,
        data.length_2,
        dyadic_order_1,
        dyadic_order_2,
        n_jobs
    )

    if err_code:
        raise Exception("Error in pysiglib.sig_kernel_backprop: " + err_msg(err_code))#

def sig_kernel_backprop_cuda_(data, derivs_data, result, gram, k_grid_data, dyadic_order_1, dyadic_order_2):
    err_code = CUSIG.batch_sig_kernel_backprop_cuda(
        cast(gram.data_ptr(), POINTER(c_double)),
        result.data_ptr,
        derivs_data.data_ptr,
        k_grid_data.data_ptr,
        data.batch_size,
        data.dimension,
        data.length_1,
        data.length_2,
        dyadic_order_1,
        dyadic_order_2
    )

    if err_code:
        raise Exception("Error in pysiglib.sig_kernel_backprop: " + err_msg(err_code))

def gram_deriv(
        derivs_data,
        data,
        gram : Union[np.ndarray, torch.tensor],
        k_grid_data : Union[np.ndarray, torch.tensor],
        dyadic_order_1,
        dyadic_order_2,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:

    result = GridOutputHandler(data.length_1 - 1, data.length_2 - 1, derivs_data) #Derivatives with respect to gram matrix

    if data.device == "cpu":
        sig_kernel_backprop_(data, derivs_data, result, gram, k_grid_data, dyadic_order_1, dyadic_order_2, n_jobs)
    else:
        if not BUILT_WITH_CUDA:
            raise RuntimeError("pySigLib was build without CUDA - data must be moved to CPU.")
        sig_kernel_backprop_cuda_(data, derivs_data, result, gram, k_grid_data, dyadic_order_1, dyadic_order_2)

    return result.data

def gram_deriv_to_path_deriv_x(
        data,
        gram_derivs : Union[np.ndarray, torch.tensor],
        y1 : Union[np.ndarray, torch.tensor],
        time_aug: bool = False,
        lead_lag: bool = False,
        end_time: float = 1.,
        n_jobs: int = 1
):
    # Now convert derivs wrt to gram into derivs wrt the path
    # note result.data is a torch array
    if data.is_batch:
        out = torch.empty((data.batch_size, data.length_1, data.length_2 - 1), dtype=torch.float64,
                          device=gram_derivs.device)
        out[:, 0, :] = 0
        out[:, 1:, :] = gram_derivs
        out[:, :-1, :] -= gram_derivs
    else:
        out = torch.empty((data.length_1, data.length_2 - 1), dtype=torch.float64, device=gram_derivs.device)
        out[0, :] = 0
        out[1:, :] = gram_derivs
        out[:-1, :] -= gram_derivs
        out = out[None, :, :]

    out = torch.bmm(out, y1)

    if lead_lag or time_aug:
        out = transform_path_backprop(out, time_aug, lead_lag, end_time, n_jobs)

    if data.type_ == "numpy":
        return out.numpy()
    return out

def gram_deriv_to_path_deriv_y(
        data,
        gram_derivs : Union[np.ndarray, torch.tensor],
        x1 : Union[np.ndarray, torch.tensor],
        time_aug: bool = False,
        lead_lag: bool = False,
        end_time: float = 1.,
        n_jobs: int = 1
):
    if data.is_batch:
        out = torch.empty((data.batch_size, data.length_1 - 1, data.length_2), dtype=torch.float64,
                          device=gram_derivs.device)
        out[:, :, 0] = 0
        out[:, :, 1:] = gram_derivs
        out[:, :, :-1] -= gram_derivs
    else:
        out = torch.empty((data.length_1 - 1, data.length_2), dtype=torch.float64, device=gram_derivs.device)
        out[:, 0] = 0
        out[:, 1:] = gram_derivs
        out[:, :-1] -= gram_derivs
        out = out[None, :, :]

    out = torch.bmm(out.permute(0,2,1), x1)

    if lead_lag or time_aug:
        out = transform_path_backprop(out, time_aug, lead_lag, end_time, n_jobs)

    if data.type_ == "numpy":
        return out.numpy()
    return out

def sig_kernel_backprop(
        derivs : Union[np.ndarray, torch.tensor],
        path1 : Union[np.ndarray, torch.tensor],
        path2 : Union[np.ndarray, torch.tensor],
        dyadic_order : Union[int, tuple],
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        left_deriv : bool = True,
        right_deriv : bool = False,
        k_grid : Union[np.ndarray, torch.tensor] = None,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor, Tuple[np.ndarray, np.ndarray], Tuple[torch.tensor, torch.tensor]]:
    """
    This function is required to backpropagate through ``pysiglib.sig_kernel``.
    Given the derivatives of a scalar function :math:`F` with respect to a
    signature kernel, :math:`\\partial F / \\left< S(x), S(y) \\right>`,
    returns the derivatives of :math:`F` with respect to one or both of the
    underlying paths, :math:`\\{\\partial F / x_{t_i}\\}_{i=0}^{L_1}` and
    :math:`\\{\\partial F / y_{t_i}\\}_{i=0}^{L_2}`.

    :param derivs: Derivatives with respect to a signature kernel or batch
        of signature kernels, :math:`\\partial F / \\left< S(x), S(y) \\right>`.
    :type derivs: numpy.ndarray | torch.tensor
    :param path1: The first underlying path or batch of paths, given as a `numpy.ndarray` or
        `torch.tensor`. For a single path, this must be of shape (length, dimension). For a
        batch of paths, this must be of shape (batch size, length, dimension).
    :type path1: numpy.ndarray | torch.tensor
    :param path2: The second underlying path or batch of paths, given as a `numpy.ndarray`
        or `torch.tensor`. For a single path, this must be of shape (length, dimension).
        For a batch of paths, this must be of shape (batch size, length, dimension).
    :type path2: numpy.ndarray | torch.tensor
    :param dyadic_order: The dyadic order(s) used to compute the signature kernels.
    :type dyadic_order: int | tuple
    :param time_aug: If ``True``, assumes the paths were time augmented.
    :type time_aug: bool
    :param lead_lag: If ``True``, assumes the lead-lag transform was applied.
    :type lead_lag: bool
    :param left_deriv: If ``True``, returns :math:`\\{\\partial F / x_{t_i}\\}_{i=0}^{L_1}`.
        At least one of ``left_deriv`` and ``right_deriv`` must be ``True``. If both are
        ``True``, returns both derivatives as a tuple.
    :type left_deriv: bool
    :param right_deriv: If ``True``, returns :math:`\\{\\partial F / y_{t_i}\\}_{i=0}^{L_2}`.
        At least one of ``left_deriv`` and ``right_deriv`` must be ``True``. If both are
        ``True``, returns both derivatives as a tuple.
    :type right_deriv: bool
    :param n_jobs: (Only applicable to CPU computation) Number of threads to run in parallel.
        If n_jobs = 1, the computation is run serially. If set to -1, all available threads
        are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example
        if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :return: Tuple of derivatives of :math:`F` with respect to one or both of the
        underlying paths. If ``left_deriv`` is ``True``, the first element of
        this tuple is  :math:`\\{\\partial F / x_{t_i}\\}_{i=0}^{L_1}`, otherwise
        it is ``None``. Similarly for ``right_deriv`` and
        :math:`\\{\\partial F / y_{t_i}\\}_{i=0}^{L_2}`.
    :rtype: numpy.ndarray | torch.tensor | Tuple[numpy.ndarray | numpy.ndarray] | Tuple[torch.tensor | torch.tensor]

    .. note::

        Ideally, any array passed to ``pysiglib.sig_kernel_backprop`` should be both contiguous and own its data.
        If this is not the case, ``pysiglib.sig_kernel_backprop`` will internally create a contiguous copy, which may be
        inefficient.
    """
    check_type(n_jobs, "n_jobs", int)
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")
    check_type(left_deriv, "left_deriv", bool)
    check_type(right_deriv, "right_deriv", bool)
    if not (left_deriv or right_deriv):
        return None, None

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

    data = DoublePathInputHandler(path1, path2, False, False, end_time, "path1", "path2")

    derivs = torch.as_tensor(derivs, dtype=torch.double)
    derivs_data = ScalarInputHandler(derivs, data.is_batch, "derivs")

    if not (derivs_data.type_ == data.type_ and derivs_data.device == data.device):
        raise ValueError("derivs, path1 and path2 must all be numpy arrays or all torch tensors on the same device")
    if data.batch_size != derivs_data.batch_size:
        raise ValueError("batch size for derivs does not match batch size of paths")

    torch_path1 = torch.as_tensor(data.path1, dtype = torch.double)  # Avoids data copy
    torch_path2 = torch.as_tensor(data.path2, dtype = torch.double)

    if k_grid is None:
        k_grid = sig_kernel(torch.as_tensor(path1), torch.as_tensor(path2), dyadic_order, False, False, end_time, n_jobs, True)

    if data.is_batch:
        x1 = torch_path1[:, 1:, :] - torch_path1[:, :-1, :]
        y1 = torch_path2[:, 1:, :] - torch_path2[:, :-1, :]
    else:
        x1 = (torch_path1[1:, :] - torch_path1[:-1, :])[None, :, :]
        y1 = (torch_path2[1:, :] - torch_path2[:-1, :])[None, :, :]

    gram = torch.empty((x1.shape[0], x1.shape[1], y1.shape[1]), dtype = torch.double, device = x1.device)
    torch.bmm(x1, y1.permute(0, 2, 1), out = gram)

    ld, rd = None, None

    k_grid_data = PathInputHandler(k_grid, False, False, 0., "k_grid")
    gram_derivs = gram_deriv(derivs_data, data, gram, k_grid_data, dyadic_order_1, dyadic_order_2, n_jobs)

    if left_deriv:
        ld = gram_deriv_to_path_deriv_x(data, gram_derivs, y1, time_aug, lead_lag, end_time, n_jobs)

    if right_deriv:
        rd = gram_deriv_to_path_deriv_y(data, gram_derivs, x1, time_aug, lead_lag, end_time, n_jobs)

    return ld, rd


