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

from copy import deepcopy
import pytest
import numpy as np
import torch

import pysiglib

np.random.seed(42)
torch.manual_seed(42)
EPSILON = 1e-2

def check_close(a, b):
    a_ = np.array(a)
    b_ = np.array(b)
    assert not np.any(np.abs(a_ - b_) > EPSILON)

def finite_difference(x1, x2, dyadic_order, time_aug = False, lead_lag = False):
    x1 = x1.to(dtype = torch.double)
    x2 = x2.to(dtype = torch.double)
    if len(x1.shape) == 2:
        x1 = x1[None, :, :]
        x2 = x2[None, :, :]
    batch_size = x1.shape[0]
    length = x1.shape[1]
    dim = x1.shape[2]

    eps = 1e-10
    k = pysiglib.sig_kernel(x1, x2, dyadic_order, time_aug = time_aug, lead_lag = lead_lag)
    out = np.empty(shape = (batch_size, length, dim))

    for i in range(length):
        for d in range(dim):
            x1_d = deepcopy(x1)
            x1_d[:,i,d] += eps
            k_d = pysiglib.sig_kernel(x1_d, x2, dyadic_order, time_aug = time_aug, lead_lag = lead_lag)
            out[:,i,d] = (k_d - k) / eps
    return out

@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_backprop_1(dyadic_order):
    X = torch.rand(size=(5, 5))
    Y = torch.rand(size=(10, 5))
    derivs = torch.ones(1)

    d1 = finite_difference(X, Y, dyadic_order)
    d2 = finite_difference(Y, X, dyadic_order)
    d3, d4 = pysiglib.sig_kernel_backprop(derivs, X, Y, dyadic_order, left_deriv = True, right_deriv = True)

    check_close(d1, d3)
    check_close(d2, d4)

@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_backprop_2(dyadic_order):
    X = torch.rand(size=(10, 5))
    Y = torch.rand(size=(5, 5))
    derivs = torch.ones(1)

    d1 = finite_difference(X, Y, dyadic_order)
    d2 = finite_difference(Y, X, dyadic_order)
    d3, d4 = pysiglib.sig_kernel_backprop(derivs, X, Y, dyadic_order, left_deriv = True, right_deriv = True)

    check_close(d1, d3)
    check_close(d2, d4)

@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_backprop_batch(dyadic_order):
    X = torch.rand(size=(32, 5, 5))
    Y = torch.rand(size=(32, 10, 5))
    derivs = torch.ones(32)

    d1 = finite_difference(X, Y, dyadic_order)
    d2 = finite_difference(Y, X, dyadic_order)
    d3, d4 = pysiglib.sig_kernel_backprop(derivs, X, Y, dyadic_order, left_deriv = True, right_deriv = True)

    check_close(d1, d3)
    check_close(d2, d4)

@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_backprop_batch_time_aug(dyadic_order):
    X = torch.rand(size=(32, 5, 5))
    Y = torch.rand(size=(32, 10, 5))
    derivs = torch.ones(32)

    d1 = finite_difference(X, Y, dyadic_order, time_aug = True)
    d2 = finite_difference(Y, X, dyadic_order, time_aug=True)
    d3, d4 = pysiglib.sig_kernel_backprop(derivs, X, Y, dyadic_order, time_aug = True, left_deriv = True, right_deriv = True)

    check_close(d1, d3)
    check_close(d2, d4)

@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_backprop_batch_lead_lag(dyadic_order):
    X = torch.rand(size=(32, 5, 2))
    Y = torch.rand(size=(32, 10, 2))
    derivs = torch.ones(32)

    d1 = finite_difference(X, Y, dyadic_order, lead_lag = True)
    d2 = finite_difference(Y, X, dyadic_order, lead_lag=True)
    d3, d4 = pysiglib.sig_kernel_backprop(derivs, X, Y, dyadic_order, lead_lag = True, left_deriv = True, right_deriv = True)

    check_close(d1, d3)
    check_close(d2, d4)

@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_backprop_batch_time_aug_lead_lag(dyadic_order):
    X = torch.rand(size=(32, 5, 2)) / 2
    Y = torch.rand(size=(32, 10, 2)) / 2
    derivs = torch.ones(32)

    d1 = finite_difference(X, Y, dyadic_order, time_aug = True, lead_lag = True)
    d2 = finite_difference(Y, X, dyadic_order, time_aug=True, lead_lag=True)
    d3, d4 = pysiglib.sig_kernel_backprop(derivs, X, Y, dyadic_order, time_aug = True, lead_lag = True, left_deriv = True, right_deriv = True)

    check_close(d1, d3)
    check_close(d2, d4)