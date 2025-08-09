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

import pytest
import numpy as np
import torch
import sigkernel

import pysiglib

np.random.seed(42)
torch.manual_seed(42)
EPSILON = 1e-10

def check_close(a, b):
    a_ = np.array(a)
    b_ = np.array(b)
    assert not np.any(np.abs(a_ - b_) > EPSILON)

def lead_lag(x):
    # A backpropagatable version of lead-lag
    lag = torch.repeat_interleave(x[:-1], repeats=2, dim=0)
    lag = torch.cat((lag, x[-1:]))
    lead = torch.repeat_interleave(x[1:], repeats=2, dim=0)
    lead = torch.cat((x[0:1], lead))
    path = torch.cat((lag, lead), dim=-1)
    return path

def batch_lead_lag(x):
    # A backpropagatable version of lead-lag
    lag = torch.repeat_interleave(x[:, :-1], repeats=2, dim=1)
    lag = torch.cat((lag, x[:, -1:]), dim=1)
    lead = torch.repeat_interleave(x[:, 1:], repeats=2, dim=1)
    lead = torch.cat((x[:, 0:1], lead), axis=1)
    path = torch.cat((lag, lead), dim=2)
    return path

def time_aug_lead_lag(x):
    # A backpropagatable version of lead-lag
    lag = torch.repeat_interleave(x[:-1], repeats=2, dim=0)
    lag = torch.cat((lag, x[-1:]))
    lead = torch.repeat_interleave(x[1:], repeats=2, dim=0)
    lead = torch.cat((x[0:1], lead))
    path = torch.cat((lag, lead), dim=-1)
    t = torch.linspace(0, path.shape[0] - 1, path.shape[0]).unsqueeze(1)
    path = torch.cat((path, t), dim =  1)
    return path

def batch_time_aug_lead_lag(x):
    # A backpropagatable version of lead-lag
    lag = torch.repeat_interleave(x[:, :-1], repeats=2, dim=1)
    lag = torch.cat((lag, x[:, -1:]), dim=1)
    lead = torch.repeat_interleave(x[:, 1:], repeats=2, dim=1)
    lead = torch.cat((x[:, 0:1], lead), axis=1)
    path = torch.cat((lag, lead), dim=2)
    t = torch.linspace(0, path.shape[1] - 1, path.shape[1]).unsqueeze(0)
    t = torch.tile(t, (path.shape[0], 1)).unsqueeze(2)
    path = torch.cat((path, t), dim=2)
    return path

def run_random(device, batch = 32, len1 = 100, len2 = 100, dim = 5):
    for _ in range(5):
        for dyadic_order in range(3):
            X = torch.tensor(np.random.uniform(size=(batch, len1, dim)), device=device)
            Y = torch.tensor(np.random.uniform(size=(batch, len2, dim)), device=device)

            static_kernel = sigkernel.LinearKernel()
            signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
            kernel1 = signature_kernel.compute_kernel(X, Y, 100)
            kernel2 = pysiglib.sig_kernel(X, Y, dyadic_order)

            check_close(kernel1.cpu(), kernel2.cpu())

def test_sig_kernel_trivial():
    X = torch.tensor([[0.]])

    kernel2 = pysiglib.sig_kernel(X, X, 0)

    check_close(torch.tensor([1.]), kernel2)

def test_sig_kernel_random_cpu():
    run_random("cpu")

def test_sig_kernel_random_cpu_non_square_1():
    run_random("cpu", len1 = 10, len2 = 100)

def test_sig_kernel_random_cpu_non_square_2():
    run_random("cpu", len2 = 10, len1 = 100)

def test_sig_kernel_different_dyadics_cpu():
    batch, len1, len2, dim = 32, 10, 100, 5
    dyadic_order = (2, 0)
    X = torch.tensor(np.random.uniform(size=(batch, len1, dim)), device="cpu")
    Y = torch.tensor(np.random.uniform(size=(batch, len2, dim)), device="cpu")

    kernel1 = pysiglib.sig_kernel(X, Y, dyadic_order)
    kernel2 = pysiglib.sig_kernel(Y, X, dyadic_order[::-1])

    check_close(kernel1.cpu(), kernel2.cpu())

@pytest.mark.skipif(not (pysiglib.BUILT_WITH_CUDA and torch.cuda.is_available()), reason="CUDA not available or disabled")
def test_sig_kernel_different_dyadics_cuda():
    batch, len1, len2, dim = 32, 10, 100, 5
    dyadic_order = (2, 0)
    X = torch.tensor(np.random.uniform(size=(batch, len1, dim)), device="cuda")
    Y = torch.tensor(np.random.uniform(size=(batch, len2, dim)), device="cuda")

    kernel1 = pysiglib.sig_kernel(X, Y, dyadic_order)
    kernel2 = pysiglib.sig_kernel(Y, X, dyadic_order[::-1])

    check_close(kernel1.cpu(), kernel2.cpu())

@pytest.mark.skipif(not (pysiglib.BUILT_WITH_CUDA and torch.cuda.is_available()), reason="CUDA not available or disabled")
def test_sig_kernel_random_cuda():
    run_random("cuda")

@pytest.mark.skipif(not (pysiglib.BUILT_WITH_CUDA and torch.cuda.is_available()), reason="CUDA not available or disabled")
def test_sig_kernel_random_cuda_non_square_1():
    run_random("cuda", len1 = 10, len2 = 100)

@pytest.mark.skipif(not (pysiglib.BUILT_WITH_CUDA and torch.cuda.is_available()), reason="CUDA not available or disabled")
def test_sig_kernel_random_cuda_non_square_2():
    run_random("cuda", len2 = 10, len1 = 100)


def test_sig_kernel_numpy1():
    x = np.array([[0, 1], [3, 2]])
    pysiglib.sig_kernel(x, x, 0)


def test_sig_kernel_numpy2():
    x = np.array([[[0, 1], [3, 2]]])
    pysiglib.sig_kernel(x, x, 0)


def test_sig_kernel_non_contiguous():
    # Make sure sig_kernel works with any form of array
    dim, length, batch = 10, 100, 32

    rand_data = torch.rand(size=(batch, length), dtype=torch.float64)[:, :, None]
    X_non_cont = rand_data.expand(-1, -1, dim)
    X = X_non_cont.clone()

    res1 = pysiglib.sig_kernel(X, X, 0)
    res2 = pysiglib.sig_kernel(X_non_cont, X_non_cont, 0)
    check_close(res1, res2)

    rand_data = np.random.normal(size=(batch, length))[:, :, None]
    X_non_cont = np.broadcast_to(rand_data, (batch, length, dim))
    X = np.array(X_non_cont)

    res1 = pysiglib.sig_kernel(X, X, 0)
    res2 = pysiglib.sig_kernel(X_non_cont, X_non_cont, 0)
    check_close(res1, res2)

@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_lead_lag(dyadic_order):
    X = torch.rand(size=(32, 50, 5)) / 100
    Y = torch.rand(size=(32, 100, 5)) / 100

    X_ll = batch_lead_lag(X).double()
    Y_ll = batch_lead_lag(Y).double()

    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    kernel1 = signature_kernel.compute_kernel(X_ll, Y_ll, 100)
    kernel2 = pysiglib.sig_kernel(X, Y, dyadic_order, lead_lag = True)

    check_close(kernel1.cpu(), kernel2.cpu())
