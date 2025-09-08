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
from ..sig import signature as sig_forward
from ..sig import sig_combine as sig_combine_forward
from ..sig_backprop import sig_backprop, sig_combine_backprop
from ..sig_kernel import sig_kernel as sig_kernel_forward
from ..sig_kernel_backprop import sig_kernel_backprop
from ..sig_kernel import sig_kernel_gram as sig_kernel_gram_forward
from ..sig_kernel_backprop import sig_kernel_gram_backprop
from ..sig_metrics import sig_score as sig_score_forward
from ..sig_metrics import expected_sig_score as expected_sig_score_forward
from ..sig_metrics import sig_mmd as sig_mmd_forward
from ..transform_path import transform_path as transform_path_forward
from ..transform_path_backprop import transform_path_backprop

from ..param_checks import check_type
from ..data_handlers import DoublePathInputHandler

class Signature(torch.autograd.Function):
    @staticmethod
    def forward(ctx, path, degree, time_aug, lead_lag, horner, n_jobs):
        sig = sig_forward(path, degree, time_aug, lead_lag, horner, n_jobs)

        ctx.save_for_backward(path, sig)
        ctx.degree = degree
        ctx.time_aug = time_aug
        ctx.lead_lag = lead_lag
        ctx.horner = horner
        ctx.n_jobs = n_jobs

        return sig

    @staticmethod
    def backward(ctx, grad_output):
        path, sig = ctx.saved_tensors
        grad = sig_backprop(path, sig, grad_output, ctx.degree, ctx.time_aug, ctx.lead_lag, ctx.n_jobs)
        return grad, None, None, None, None, None

def signature(
        path : Union[np.ndarray, torch.tensor],
        degree : int,
        time_aug : bool = False,
        lead_lag : bool = False,
        horner : bool = True,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    return Signature.apply(path, degree, time_aug, lead_lag, horner, n_jobs)


signature.__doc__ = sig_forward.__doc__

class SigCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sig1, sig2, dimension ,degree, n_jobs):
        sig_combined = sig_combine_forward(sig1, sig2, dimension ,degree, n_jobs)

        ctx.save_for_backward(sig1, sig2)
        ctx.dimension = dimension
        ctx.degree = degree
        ctx.n_jobs = n_jobs

        return sig_combined

    @staticmethod
    def backward(ctx, grad_output):
        sig1, sig2 = ctx.saved_tensors
        sig1_grad, sig2_grad = sig_combine_backprop(grad_output, sig1, sig2, ctx.dimension, ctx.degree, ctx.n_jobs)
        return sig1_grad, sig2_grad, None, None, None

def sig_combine(
        sig1 : Union[np.ndarray, torch.tensor],
        sig2 : Union[np.ndarray, torch.tensor],
        dimension : int,
        degree : int,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    return SigCombine.apply(sig1, sig2, dimension ,degree, n_jobs)


sig_combine.__doc__ = sig_combine_forward.__doc__

class TransformPath(torch.autograd.Function):
    @staticmethod
    def forward(ctx, path, time_aug, lead_lag, end_time, n_jobs):
        new_path = transform_path_forward(path, time_aug, lead_lag, end_time, n_jobs)

        ctx.time_aug = time_aug
        ctx.lead_lag = lead_lag
        ctx.end_time = end_time
        ctx.n_jobs = n_jobs

        return new_path

    @staticmethod
    def backward(ctx, grad_output):
        new_derivs = transform_path_backprop(grad_output, ctx.time_aug, ctx.lead_lag, ctx.end_time, ctx.n_jobs)
        return new_derivs, None, None, None, None

def transform_path(
    path : Union[np.ndarray, torch.tensor],
    time_aug : bool = False,
    lead_lag : bool = False,
    end_time : float = 1.,
    n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    return TransformPath.apply(path, time_aug, lead_lag, end_time, n_jobs)

transform_path.__doc__ = transform_path_forward.__doc__

class SigKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, path1, path2, dyadic_order, time_aug, lead_lag, end_time, n_jobs):
        k_grid = sig_kernel_forward(path1, path2, dyadic_order, time_aug, lead_lag, end_time, n_jobs, True)

        ctx.save_for_backward(k_grid, path1, path2)
        ctx.dyadic_order = dyadic_order
        ctx.time_aug = time_aug
        ctx.lead_lag = lead_lag
        ctx.end_time = end_time
        ctx.n_jobs = n_jobs

        if len(k_grid.shape) == 3:
            return k_grid[:, -1, -1]
        else:
            return k_grid[-1, -1]

    @staticmethod
    def backward(ctx, grad_output):
        left_deriv = ctx.needs_input_grad[0]
        right_deriv = ctx.needs_input_grad[1]

        k_grid, path1, path2 = ctx.saved_tensors
        new_derivs = sig_kernel_backprop(grad_output, path1, path2, ctx.dyadic_order,
                                         ctx.time_aug, ctx.lead_lag, ctx.end_time,
                                         left_deriv, right_deriv, k_grid, ctx.n_jobs)

        return new_derivs[0], new_derivs[1], None, None, None, None, None, None, None

def sig_kernel(
        path1 : Union[np.ndarray, torch.tensor],
        path2 : Union[np.ndarray, torch.tensor],
        dyadic_order : Union[int, tuple],
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    return SigKernel.apply(path1, path2, dyadic_order, time_aug, lead_lag, end_time, n_jobs)

sig_kernel.__doc__ = sig_kernel_forward.__doc__

class SigKernelGram(torch.autograd.Function):
    @staticmethod
    def forward(ctx, path1, path2, dyadic_order, time_aug, lead_lag, end_time, n_jobs, max_batch, save_kernel):
        k_grid = sig_kernel_gram_forward(path1, path2, dyadic_order, time_aug, lead_lag, end_time, n_jobs, max_batch, return_grid = save_kernel)

        if save_kernel:
            ctx.save_for_backward(k_grid, path1, path2)
        else:
            ctx.save_for_backward(path1, path2)

        ctx.dyadic_order = dyadic_order
        ctx.time_aug = time_aug
        ctx.lead_lag = lead_lag
        ctx.end_time = end_time
        ctx.n_jobs = n_jobs
        ctx.max_batch = max_batch
        ctx.save_kernel = save_kernel

        if save_kernel:
            return k_grid[:, -1, -1]
        else:
            return k_grid

    @staticmethod
    def backward(ctx, grad_output):
        left_deriv = ctx.needs_input_grad[0]
        right_deriv = ctx.needs_input_grad[1]

        if ctx.save_kernel:
            k_grid, path1, path2 = ctx.saved_tensors
        else:
            k_grid = None
            path1, path2 = ctx.saved_tensors

        new_derivs = sig_kernel_gram_backprop(grad_output, path1, path2, ctx.dyadic_order,
                                         ctx.time_aug, ctx.lead_lag, ctx.end_time,
                                         left_deriv, right_deriv, k_grid, ctx.n_jobs, ctx.max_batch)

        return new_derivs[0], new_derivs[1], None, None, None, None, None, None, None

def sig_kernel_gram(
        path1: Union[np.ndarray, torch.tensor],
        path2: Union[np.ndarray, torch.tensor],
        dyadic_order: Union[int, tuple],
        time_aug: bool = False,
        lead_lag: bool = False,
        end_time: float = 1.,
        n_jobs: int = 1,
        max_batch: int = -1,
        save_kernel: bool = False
) -> Union[np.ndarray, torch.tensor]:
    return SigKernelGram.apply(path1, path2, dyadic_order, time_aug, lead_lag, end_time, n_jobs, max_batch, save_kernel)

sig_kernel_gram.__doc__ = sig_kernel_gram_forward.__doc__

def sig_score(
        sample : Union[np.ndarray, torch.tensor],
        y : Union[np.ndarray, torch.tensor],
        dyadic_order : Union[int, tuple],
        lam : float = 1.,
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        n_jobs : int = 1,
        max_batch : int = -1
) -> Union[np.ndarray, torch.tensor]:
    check_type(sample, "sample", torch.Tensor)
    check_type(y, "y", torch.Tensor)

    # Use torch for simplicity
    sample = torch.as_tensor(sample)
    y = torch.as_tensor(y)
    if len(y.shape) == 2:
        y = y.unsqueeze(0).contiguous().clone()

    data = DoublePathInputHandler(sample, y, time_aug, lead_lag, end_time, "sample_paths", "y", True, False)

    B = sample.shape[0]

    xx = sig_kernel_gram(sample, sample, dyadic_order, time_aug, lead_lag, end_time, n_jobs, max_batch, False)
    xy = sig_kernel_gram(sample, y, dyadic_order, time_aug, lead_lag, end_time, n_jobs, max_batch, False)

    xx_sum = float((torch.sum(xx) - torch.sum(torch.diag(xx))) / (B * (B - 1)))
    xy_sum = torch.sum(xy, dim=0) * (2. / B)

    res = lam * torch.full(xy_sum.shape, xx_sum, device=xy_sum.device) - xy_sum
    return res

sig_score.__doc__ = sig_score_forward.__doc__

def expected_sig_score(
        sample1 : Union[np.ndarray, torch.tensor],
        sample2 : Union[np.ndarray, torch.tensor],
        dyadic_order : Union[int, tuple],
        lam : float = 1.,
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        n_jobs : int = 1,
        max_batch : int = -1
) -> Union[np.ndarray, torch.tensor]:
    res = sig_score(sample1, sample2, dyadic_order, lam, time_aug, lead_lag, end_time, n_jobs, max_batch)
    res = torch.mean(res, 0, True)
    return res

expected_sig_score.__doc__ = expected_sig_score_forward.__doc__

def sig_mmd(
        sample1 : Union[np.ndarray, torch.tensor],
        sample2 : Union[np.ndarray, torch.tensor],
        dyadic_order : Union[int, tuple],
        lam : float = 1.,
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        n_jobs : int = 1,
        max_batch : int = -1
) -> Union[np.ndarray, torch.tensor]:
    data = DoublePathInputHandler(sample1, sample2, time_aug, lead_lag, end_time, "sample1", "sample2", True, False)

    # Use torch for simplicity
    sample1 = torch.as_tensor(data.path1)
    sample2 = torch.as_tensor(data.path2)

    m = sample1.shape[0]
    n = sample2.shape[0]

    xx = sig_kernel_gram(sample1, sample1, dyadic_order, time_aug, lead_lag, end_time, n_jobs, max_batch, False)
    xy = sig_kernel_gram(sample1, sample2, dyadic_order, time_aug, lead_lag, end_time, n_jobs, max_batch, False)
    yy = sig_kernel_gram(sample2, sample2, dyadic_order, time_aug, lead_lag, end_time, n_jobs, max_batch, False)

    xx_sum = (torch.sum(xx) - torch.sum(torch.diag(xx))) / (m * (m - 1))
    xy_sum = 2. * torch.mean(xy)
    yy_sum = (torch.sum(yy) - torch.sum(torch.diag(yy))) / (n * (n - 1))

    return xx_sum - xy_sum + yy_sum

sig_mmd.__doc__ = sig_mmd_forward.__doc__
