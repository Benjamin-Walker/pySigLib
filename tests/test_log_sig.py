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

try:
    import signatory
except:
    signatory = None

import pysiglib

np.random.seed(42)
torch.manual_seed(42)

SINGLE_EPSILON = 1e-4
DOUBLE_EPSILON = 1e-10

def check_close(a, b):
    a_ = np.array(a)
    b_ = np.array(b)
    EPSILON = SINGLE_EPSILON if a_.dtype == np.float32 else DOUBLE_EPSILON
    assert not np.any(np.abs(a_ - b_) > EPSILON)

@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_log_signature_expanded_random(deg, dtype):
    X = np.random.uniform(size=(100, 5)).astype(dtype)

    s = iisignature.prepare(5, deg, "x")
    iisig = iisignature.logsig(X, s, "x").astype(dtype)
    pysiglib.prepare_log_sig(5, deg)
    sig = pysiglib.log_sig(X, deg, method=0)
    check_close(iisig, sig[1:])
    pysiglib.reset_log_sig()

@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_batch_log_signature_expanded_random(deg, dtype):
    X = np.random.uniform(size=(32, 100, 5)).astype(dtype)

    s = iisignature.prepare(5, deg, "x")
    iisig = iisignature.logsig(X, s, "x").astype(dtype)
    pysiglib.prepare_log_sig(5, deg)
    sig = pysiglib.log_sig(X, deg, method=0)
    check_close(iisig, sig[:, 1:])
    pysiglib.reset_log_sig()

@pytest.mark.skipif(signatory is None, reason="signatory not available")
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_log_signature_lyndon_words_random(deg, dtype):
    X = torch.rand(size=(1, 100, 5), dtype=dtype)

    ls = signatory.logsignature(X, deg, mode="words")[0]
    pysiglib.prepare_log_sig(5, deg)
    sig = pysiglib.log_sig(X[0], deg, method=1)
    check_close(ls, sig)
    pysiglib.reset_log_sig()

@pytest.mark.skipif(signatory is None, reason="signatory not available")
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_batch_log_signature_lyndon_words_random(deg, dtype):
    X = torch.rand(size=(32, 100, 5), dtype=dtype)

    ls = signatory.logsignature(X, deg, mode="words")
    pysiglib.prepare_log_sig(5, deg)
    sig = pysiglib.log_sig(X, deg, method=1)
    check_close(ls, sig)
    pysiglib.reset_log_sig()

@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_log_signature_lyndon_basis_random(deg, dtype):
    X = np.random.uniform(size=(100, 5)).astype(dtype)

    s = iisignature.prepare(5, deg, "s")
    iisig = iisignature.logsig(X, s, "s").astype(dtype)
    pysiglib.prepare_log_sig(5, deg)
    sig = pysiglib.log_sig(X, deg, method=2)
    check_close(iisig, sig)
    pysiglib.reset_log_sig()

@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_batch_log_signature_lyndon_basis_random(deg, dtype):
    X = np.random.uniform(size=(32, 100, 5)).astype(dtype)

    s = iisignature.prepare(5, deg, "s")
    iisig = iisignature.logsig(X, s, "s").astype(dtype)
    pysiglib.prepare_log_sig(5, deg)
    sig = pysiglib.log_sig(X, deg, method=2)
    check_close(iisig, sig)
    pysiglib.reset_log_sig()
