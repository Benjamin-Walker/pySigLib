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
