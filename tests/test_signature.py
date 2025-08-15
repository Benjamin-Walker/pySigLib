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
import iisignature

import pysiglib

np.random.seed(42)
torch.manual_seed(42)
EPSILON = 1e-10


def lead_lag(X):
    lag = []
    lead = []

    for val_lag, val_lead in zip(X[:-1], X[1:]):
        lag.append(val_lag)
        lead.append(val_lag)

        lag.append(val_lag)
        lead.append(val_lead)

    lag.append(X[-1])
    lead.append(X[-1])

    return np.c_[lag, lead]

def check_close(a, b):
    a_ = np.array(a)
    b_ = np.array(b)
    assert not np.any(np.abs(a_ - b_) > EPSILON)

def test_signature_trivial():
    check_close(pysiglib.signature(np.array([[0, 0], [1, 1]]), 0), [1.])
    check_close(pysiglib.signature(np.array([[0, 0], [1, 1]]), 1), [1., 1., 1.])
    check_close(pysiglib.signature(np.array([[0, 0]]), 1), [1., 0., 0.])


@pytest.mark.parametrize("deg", range(1, 6))
def test_signature_random(deg):
    X = np.random.uniform(size=(100, 5))
    iisig = iisignature.sig(X, deg)
    sig = pysiglib.signature(X, deg)
    check_close(iisig, sig[1:])


@pytest.mark.parametrize("deg", range(1, 6))
def test_signature_random_batch(deg):
    X = np.random.uniform(size=(32, 100, 5))
    iisig = iisignature.sig(X, deg)
    sig_serial = pysiglib.signature(X, deg, n_jobs=1)
    sig_parallel = pysiglib.signature(X, deg, n_jobs=-1)
    check_close(iisig, sig_serial[:, 1:])
    check_close(iisig, sig_parallel[:, 1:])


@pytest.mark.parametrize("deg", range(1, 6))
def test_signature_random_int(deg):
    X = np.random.randint(-2, 2, size=(100, 5))
    iisig = iisignature.sig(X, deg)
    sig = pysiglib.signature(X, deg)
    check_close(iisig, sig[1:])


@pytest.mark.parametrize("deg", range(1, 6))
def test_signature_random_int_batch(deg):
    X = np.random.randint(-2, 2, size=(32, 100, 5))
    iisig = iisignature.sig(X, deg)
    sig_serial = pysiglib.signature(X, deg, n_jobs=1)
    sig_parallel = pysiglib.signature(X, deg, n_jobs=-1)
    check_close(iisig, sig_serial[:, 1:])
    check_close(iisig, sig_parallel[:, 1:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_signature_cuda_err():
    x = torch.tensor([[0.], [1.]], device="cuda")
    with pytest.raises(ValueError):
        pysiglib.signature(x, 2)


def test_signature_non_contiguous():
    # Make sure signature works with any form of array
    dim, degree, length, batch = 10, 3, 100, 32

    rand_data = torch.rand((batch, length), dtype=torch.float64)[:, :, None]
    X_non_cont = rand_data.expand(-1, -1, dim)
    X = X_non_cont.clone()

    res1 = pysiglib.signature(X, degree)
    res2 = pysiglib.signature(X_non_cont, degree)
    check_close(res1, res2)

    rand_data = np.random.normal(size=(batch, length))[:, :, None]
    X_non_cont = np.broadcast_to(rand_data, (batch, length, dim))
    X = np.array(X_non_cont)

    res1 = pysiglib.signature(X, degree)
    res2 = pysiglib.signature(X_non_cont, degree)
    check_close(res1, res2)

@pytest.mark.parametrize("deg", range(1, 6))
def test_signature_time_aug(deg):
    X = np.random.uniform(size=(10, 4))
    t = np.linspace(0, 1, 10)[:, np.newaxis]
    X_aug = np.concatenate([X, t], axis = 1)
    iisig = iisignature.sig(X_aug, deg)
    sig = pysiglib.signature(X, deg, time_aug = True)
    check_close(iisig, sig[1:])

@pytest.mark.parametrize("deg", range(1, 6))
def test_signature_lead_lag(deg):
    X = np.random.uniform(size=(10, 2))
    X_aug = lead_lag(X)
    iisig = iisignature.sig(X_aug, deg)
    sig = pysiglib.signature(X, deg, lead_lag = True)
    check_close(iisig, sig[1:])

@pytest.mark.parametrize("deg", range(1, 6))
def test_signature_time_aug_lead_lag(deg):
    X = np.random.uniform(size=(10, 2))
    X_aug = lead_lag(X)
    t = np.linspace(0, 1, 19)[:, np.newaxis]
    X_aug = np.concatenate([X_aug, t], axis = 1)
    iisig = iisignature.sig(X_aug, deg)
    sig = pysiglib.signature(X, deg, lead_lag = True, time_aug = True)
    check_close(iisig, sig[1:])

#TODO: add time_aug and lead_lag tests