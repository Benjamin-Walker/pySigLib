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

SINGLE_EPSILON = 1e-4
DOUBLE_EPSILON = 1e-10

def check_close(a, b):
    a_ = np.array(a)
    b_ = np.array(b)
    EPSILON = SINGLE_EPSILON if a_.dtype == np.float32 else DOUBLE_EPSILON
    assert not np.any(np.abs(a_ - b_) > EPSILON)

@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("method", ["x"])
def test_log_signature_random(deg, dtype, method):
    X = np.random.uniform(size=(100, 5)).astype(dtype)

    s = iisignature.prepare(5, deg, method)
    iisig = iisignature.logsig(X, s, method).astype(dtype)
    sig = pysiglib.log_sig(X, deg, method=method).astype(dtype)
    check_close(iisig, sig[1:])

@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("method", ["x"])
def test_batch_log_signature_x_random(deg, dtype, method):
    X = np.random.uniform(size=(32, 100, 5)).astype(dtype)

    s = iisignature.prepare(5, deg, method)
    iisig = iisignature.logsig(X, s, method).astype(dtype)
    sig = pysiglib.log_sig(X, deg, method=method).astype(dtype)
    check_close(iisig, sig[:, 1:])
