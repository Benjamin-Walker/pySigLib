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

import pysiglib

np.random.seed(42)
torch.manual_seed(42)
EPSILON = 1e-10


def check_close(a, b):
    a_ = np.array(a)
    b_ = np.array(b)
    assert not np.any(np.abs(a_ - b_) > EPSILON)

@pytest.mark.parametrize("deg", range(1, 6))
def test_sig_combine_random(deg):
    X1 = np.random.uniform(size=(100, 5))
    X2 = np.random.uniform(size=(100, 5))
    X = np.concatenate((X1, X2), axis=0)
    X2 = np.concatenate((X1[[-1], :], X2), axis=0)
    sig1 = pysiglib.signature(X1, deg)
    sig2 = pysiglib.signature(X2, deg)
    sig = pysiglib.signature(X, deg)
    sig_mult = pysiglib.sig_combine(sig1, sig2, 5, deg)
    check_close(sig, sig_mult)


@pytest.mark.parametrize("deg", range(1, 6))
def test_sig_combine_random_batch(deg):
    X1 = np.random.uniform(size=(32, 100, 5))
    X2 = np.random.uniform(size=(32, 100, 5))
    X = np.concatenate((X1, X2), axis=1)
    X2 = np.concatenate((X1[:, [-1], :], X2), axis=1)
    sig1 = pysiglib.signature(X1, deg)
    sig2 = pysiglib.signature(X2, deg)
    sig = pysiglib.signature(X, deg)
    sig_mult = pysiglib.sig_combine(sig1, sig2, 5, deg)
    check_close(sig, sig_mult)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sig_combine_cuda_err():
    x = torch.tensor([[0.]], dtype=torch.float64, device="cuda")
    with pytest.raises(ValueError):
        pysiglib.sig_combine(x, x, 1, 0)


def test_sig_combine_non_contiguous():
    dim, degree, batch = 10, 3, 32
    sig_length = pysiglib.sig_length(dim, degree)

    rand_data = torch.rand(size=(batch,), dtype=torch.float64)[:, None]
    X_non_cont = rand_data.expand(-1, sig_length)
    X = X_non_cont.clone()

    res1 = pysiglib.sig_combine(X, X, dim, degree)
    res2 = pysiglib.sig_combine(X_non_cont, X_non_cont, dim, degree)
    check_close(res1, res2)

    rand_data = np.random.normal(size=batch)[:, None]
    X_non_cont = np.broadcast_to(rand_data, (batch, sig_length))
    X = np.array(X_non_cont)

    res1 = pysiglib.sig_combine(X, X, dim, degree)
    res2 = pysiglib.sig_combine(X_non_cont, X_non_cont, dim, degree)
    check_close(res1, res2)