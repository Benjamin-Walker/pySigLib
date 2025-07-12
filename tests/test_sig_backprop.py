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
EPSILON = 1e-5

def check_close(a, b):
    a_ = np.array(a)
    b_ = np.array(b)
    assert not np.max(np.abs(a_ - b_)) > EPSILON


@pytest.mark.parametrize("deg", range(1, 6))
def test_sig_backprop_random(deg):
    X = np.random.uniform(size=(100, 5))
    sig_derivs = np.random.uniform(size = pysiglib.sig_length(5, deg))

    sig = pysiglib.signature(X, deg)

    sig_back1 = pysiglib.sig_backprop(X.copy(), sig.copy(), sig_derivs.copy(), deg)
    sig_back2 = iisignature.sigbackprop(sig_derivs[1:].copy(), X.copy(), deg)
    check_close(sig_back1, sig_back2)

@pytest.mark.parametrize("deg", range(1, 6))
def test_batch_sig_backprop_random(deg):
    X = np.random.uniform(size=(100, 3, 2)).astype("double")
    sig_derivs = np.random.uniform(size = (100, pysiglib.sig_length(2, deg))).astype("double")

    sig = pysiglib.signature(X.copy(), deg)

    sig_back1 = pysiglib.sig_backprop(X.copy(), sig.copy(), sig_derivs.copy(), deg)
    sig_back2 = iisignature.sigbackprop(sig_derivs[:, 1:].copy(), X.copy(), deg)
    check_close(sig_back1, sig_back2)
