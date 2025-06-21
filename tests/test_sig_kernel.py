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

def run_random(device):
    for _ in range(5):
        for dyadic_order in range(3):
            X = torch.tensor(np.random.uniform(size=(32, 50, 5)), device=device)
            Y = torch.tensor(np.random.uniform(size=(32, 100, 5)), device=device)

            static_kernel = sigkernel.LinearKernel()
            signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
            kernel1 = signature_kernel.compute_kernel(X, Y, 100)
            kernel2 = pysiglib.sig_kernel(X, Y, dyadic_order)

            check_close(kernel1.cpu(), kernel2.cpu())


def test_sig_kernel_random_cpu():
    run_random("cpu")


@pytest.mark.skipif(not (pysiglib.BUILT_WITH_CUDA and torch.cuda.is_available()), reason="CUDA not available or disabled")
def test_sig_kernel_random_cuda():
    run_random("cuda")


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