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

import timeit

import numpy as np
import torch
import iisignature
# import esig
import signatory
from tqdm import tqdm

import pysiglib

# def esigbatch(X_, deg):
#     for i in range(X_.shape[0]):
#         esig.stream2sig(X_[i], deg)

# length = 1024
# dimension = 16
# degree = 4
# batch_size = 32

length = 128
dimension = 4
degree = 7
batch_size = 32

num_runs = 50

X = np.random.uniform(size=(batch_size, length, dimension)).astype("double")

def time_function(f, *args, **kwargs):
    best_time = float('inf')
    for _ in tqdm(range(num_runs)):
        start = timeit.default_timer()
        f(*args, **kwargs)
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time

if __name__ == '__main__':
    #print("\nesig (serial): ", timeFunction(esigbatch, X, degree))
    print("\niisignature (serial): ", time_function(iisignature.sig, X, degree))
    print("\npysiglib (serial): ", time_function(pysiglib.signature, X, degree, parallel = False))
    #
    #
    print("\nsignatory (parallel): ", time_function(signatory.signature, torch.tensor(X), degree))
    print("\npysiglib (parallel): ", time_function(pysiglib.signature, X, degree, parallel=True))
