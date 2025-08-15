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
EPSILON = 1e-5 #TODO: not an exact match with iisig

def check_close(a, b):
    a_ = np.array(a)
    b_ = np.array(b)
    assert not np.max(np.abs(a_ - b_)) > EPSILON

def time_aug(X, end_time = 1., is_batch = False):
    length = X.shape[1] if is_batch else X.shape[0]
    batch_size = X.shape[0] if is_batch else None
    t = np.linspace(0, end_time, length)
    t = np.tile(t[np.newaxis, :, np.newaxis], (batch_size, 1, 1)) if is_batch else t[:, np.newaxis]
    return np.concatenate((X, t), axis = 2 if is_batch else 1)

def lead_lag(X, is_batch = False):
    if not is_batch:
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
    else:
        return np.concatenate([lead_lag(X_)[np.newaxis, :, :] for X_ in X], axis = 0)

def test_transform_path_time_aug():
    X = np.random.uniform(size=(100, 5))
    X1 = time_aug(X)
    X2 = pysiglib.transform_path(X, time_aug = True)
    check_close(X1, X2)

def test_batch_transform_path_time_aug():
    X = np.random.uniform(size=(10, 100, 5))
    X1 = time_aug(X, is_batch = True)
    X2 = pysiglib.transform_path(X, time_aug = True)
    check_close(X1, X2)

def test_transform_path_lead_lag():
    X = np.random.uniform(size=(100, 5))
    X1 = lead_lag(X)
    X2 = pysiglib.transform_path(X, lead_lag = True)
    check_close(X1, X2)

def test_batch_transform_path_lead_lag():
    X = np.random.uniform(size=(10, 100, 5))
    X1 = lead_lag(X, True)
    X2 = pysiglib.transform_path(X, lead_lag = True)
    check_close(X1, X2)

