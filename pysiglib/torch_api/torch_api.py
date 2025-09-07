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
from ..transform_path import transform_path as transform_path_forward
from ..transform_path_backprop import transform_path_backprop

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
